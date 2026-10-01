"""Prime Agent sessions as a Right Rudder op stream.

load_session(path) walks the root session along its live branch, finds the child sessions it spawned, and
merges every stream into one op list in time order. Each op's `source` names where it came from: `root`,
`child:<name>`, or `compaction`.

Mapping:
  * a git_state delta that adds a file      -> add  file <path>
  * a git_state delta that changes a file   -> set  file <path>
  * a write an ipython cell performed       -> set  <kind> <key>   (files by default; more via write_patterns)
    a cell whose kernel status is "error" reads as rejected; where a git_state snapshot lists files, a
    claimed file write the next snapshot never shows also reads as rejected
    (Prime Agent's own git_state records commit and branch only, so that check is inert on its sessions)
  * an agent_message a session receives     -> answer ops citing the files and named facts it mentions
  * a message a child delivers              -> set  report "<child>: <message>"
  * an "[agent-message from <child>]" header in an orchestrator's own text, with no delivery from that child
    carrying the same message anywhere in the session
                                            -> answer citing report "<child>: <message>", which the read
                                               flags: the text claims a message that was never delivered
  * a compaction summary                    -> answer ops from `compaction` citing what its current-state
                                               section states (history in the summary is not a claim)
  * the task a child received from the root -> answer ops from `root` citing what the root handed over
  * a child's exit notice                   -> answer ops from that child
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

try:
    from right_rudder.ops import Op
except ImportError:  # pragma: no cover
    from dataclasses import asdict

    @dataclass
    class Op:  # the Right Rudder op contract, for use without right-rudder installed
        op: str
        kind: str
        key: str
        value: Optional[str] = None
        to: Optional[str] = None
        ok: bool = True
        refs: List[Tuple[str, str]] = field(default_factory=list)
        step: Optional[int] = None
        source: Optional[str] = None

        def as_dict(self) -> Dict[str, Any]:
            d = asdict(self)
            d["refs"] = [list(r) for r in self.refs]
            return d

AGENT_MESSAGE = "agent_message"
REPORT = "report"
# the header Prime Agent puts on a delivered agent message; the same text in an agent's own output is a claim
HEADER = re.compile(r"\[agent-message from (?:child:)?([^\]\s]+)\]")
RECEIVED = (AGENT_MESSAGE, "rlm_child_terminal_notice")
IPYTHON = "ipython"
GIT_FILE_FIELDS = ("files", "changedFiles", "changes", "status", "dirtyFiles", "entries")

# File writes a cell can perform. Each pattern's first group is the path.
FILE_WRITE_PATTERNS = [
    r"""open\(\s*[rbfu]?['"]([^'"]+)['"]\s*,\s*[rbfu]?['"][wax]""",
    r"""Path\(\s*[rbfu]?['"]([^'"]+)['"]\s*\)\s*\.write_(?:text|bytes)\(""",
    r"""(?:write_file|edit_file|replace_in_file|apply_edit)\(\s*[rbfu]?['"]([^'"]+)['"]""",
]
FILE_REF = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,8})(?![\w/-])")


@dataclass
class WritePattern:
    """A non-file write a cell can perform, e.g. a skill call. Named groups `key` and `value`."""
    regex: str
    kind: str

    def compiled(self) -> Pattern[str]:
        return re.compile(self.regex)


@dataclass
class _Timed:
    ts: str
    order: int
    op: Op


def _read_jsonl(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    header: Dict[str, Any] = {}
    entries: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "session" and not header:
                header = d
            elif "id" in d:
                entries.append(d)
    return header, entries


def live_branch(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Entries from the root to the leaf (the last entry written), in order."""
    if not entries:
        return []
    by_id = {e["id"]: e for e in entries}
    path, cur = [], entries[-1]
    seen = set()
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        path.append(cur)
        cur = by_id.get(cur.get("parentId"))
    return list(reversed(path))


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _cell_code(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        for k in ("code", "source", "cell", "input"):
            if isinstance(arguments.get(k), str):
                return arguments[k]
        for v in arguments.values():
            if isinstance(v, str):
                return v
    return ""


class _Resolver:
    """Finds the files and named facts a piece of text mentions."""

    def __init__(self, key_pattern: Optional[str], kind: str, qualify: bool = False):
        self.kind = kind
        self.qualify = qualify
        self.key_re = re.compile(key_pattern) if key_pattern else None
        self.fact_re = (re.compile(r"(?<![\w.])['\"]?(" + key_pattern + r")['\"]?\s*(?:=|:|is|->|→)\s*(-?\d+(?:\.\d+)?|[\w./-]+)")
                        if key_pattern else None)

    def ops(self, text: str, known_files: Iterable[str], source: str) -> List[Op]:
        out: List[Op] = []
        seen = set()
        if self.fact_re is not None:
            for m in self.fact_re.finditer(text):
                key, val = m.group(1), m.group(m.lastindex)
                if (key, val) in seen or self.key_re.fullmatch(val):
                    continue
                seen.add((key, val))
                out.append(Op("answer", self.kind, key, value=f"{key}={val}" if self.qualify else val,
                              refs=[(self.kind, key)], source=source))
            named = {k for k, _ in seen}
            for m in self.key_re.finditer(text):
                key = m.group(0)
                if key not in named:
                    named.add(key)
                    out.append(Op("answer", self.kind, key, value=None, refs=[(self.kind, key)], source=source))
        files = set(known_files)
        for m in FILE_REF.finditer(text):
            p = m.group(1)
            if p in files or "/" in p:
                if ("file", p) not in seen:
                    seen.add(("file", p))
                    out.append(Op("answer", "file", p, value=None, refs=[("file", p)], source=source))
        return out


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def message_parts(text: str) -> List[Tuple[str, str]]:
    """(sender, message) for every "[agent-message from <sender>]" header in a text: the message is the first
    paragraph after the header."""
    out: List[Tuple[str, str]] = []
    for m in HEADER.finditer(text or ""):
        rest = text[m.end():].lstrip()
        nxt = HEADER.search(rest)
        body = rest[:nxt.start()] if nxt else rest
        body = re.split(r"\n\s*\n", body.strip(), maxsplit=1)[0] if body.strip() else ""
        if body.strip():
            out.append((m.group(1), body.strip()))
    return out


def _delivered_by(details: Any) -> Optional[str]:
    """The child that sent a delivered message, or None when it came from the parent."""
    if not isinstance(details, dict) or details.get("fromRelationship") != "child":
        return None
    s = details.get("from") or details.get("sender")
    if isinstance(s, dict) and s.get("sessionName"):
        return str(s["sessionName"])
    name = details.get("senderName") or details.get("childName")
    return str(name) if name else None


def deliveries(branch: List[Dict[str, Any]]) -> set:
    """{(sender, normalised message)} for every message a child delivered on this branch."""
    out = set()
    for e in branch:
        content, details = None, None
        if e.get("type") == "custom_message" and e.get("customType") == AGENT_MESSAGE:
            content, details = e.get("content"), e.get("details")
        elif e.get("type") == "message" and (e.get("message") or {}).get("customType") == AGENT_MESSAGE:
            content, details = e["message"].get("content"), e["message"].get("details")
        who = _delivered_by(details)
        if who is None:
            continue
        parts = message_parts(_text(content)) or [(who, _text(content).strip())]
        for _, body in parts:
            out.add((who, _norm(body)))
    return out


_CURRENT_LABEL = re.compile(r"^(\s*)(#+\s*|[-*]\s+)?\**\s*current\b[^\n]{0,40}?\b(?:state|values?)\b", re.I)


def current_state_section(summary: str) -> str:
    """The parts of a compaction summary it presents as the current state: text after a label such as "Current
    ledger state:" or a "## Current state" heading, up to the end of that list or section. Values the summary
    gives as history ("Completed round 0: e11 updated to 73") are not claims about the present."""
    lines = (summary or "").split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        m = _CURRENT_LABEL.match(lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        mark = (m.group(2) or "").strip()
        level = len(mark) if mark.startswith("#") else None
        out.append(lines[i].split(":", 1)[1] if ":" in lines[i] else "")
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if level is not None:
                h = re.match(r"^\s*(#+)\s", line)
                if h and len(h.group(1)) <= level:
                    break
            else:
                if not line.strip() or len(line) - len(line.lstrip()) <= indent:
                    break
            out.append(line)
            j += 1
        i = j
    return "\n".join(out)


def _git_paths(git: Any) -> Dict[str, str]:
    """{path: status} from a git_state snapshot, tolerant of field names."""
    out: Dict[str, str] = {}
    if not isinstance(git, dict):
        return out
    for field_name in GIT_FILE_FIELDS:
        v = git.get(field_name)
        if isinstance(v, list):
            for f in v:
                if isinstance(f, str):
                    out[f] = "M"
                elif isinstance(f, dict):
                    p = f.get("path") or f.get("file") or f.get("name")
                    if p:
                        out[str(p)] = str(f.get("status") or f.get("state") or f.get("change") or "M")
        elif isinstance(v, dict):
            for p, s in v.items():
                out[str(p)] = str(s)
    for name in ("head", "commit", "sha", "branch"):
        if isinstance(git.get(name), str):
            out.setdefault("\0" + name, git[name])
    return out


def _session_ops(path: str, source: str, resolver: _Resolver, writes: List[Tuple[Pattern[str], str]],
                 known_files: set) -> Tuple[List[_Timed], List[str]]:
    """Ops from one session's live branch, plus spawn handles found in its cell outputs."""
    header, entries = _read_jsonl(path)
    branch = live_branch(entries)
    delivered = deliveries(branch)
    out: List[_Timed] = []
    handles: List[str] = []
    pending: Dict[str, str] = {}
    claimed: List[Tuple[int, str]] = []          # (index into out, path) of cell-claimed file writes
    prev_git: Optional[Dict[str, str]] = None
    order = 0

    def emit(ts: str, op: Op) -> int:
        nonlocal order
        out.append(_Timed(ts, order, op))
        order += 1
        return len(out) - 1

    for e in branch:
        ts = e.get("timestamp", "")
        t = e.get("type")
        if t == "message":
            m = e.get("message", {})
            role = m.get("role")
            if role == "assistant":
                for part in m.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "toolCall" and part.get("name") == IPYTHON:
                        pending[part.get("id", "")] = _cell_code(part.get("arguments"))
                # an agent-message header in the agent's own text claims a delivery; unless a delivery from that
                # sender carries the same message somewhere in the session, the claim cites a report never set
                for who, body in message_parts(_text(m.get("content"))):
                    if (who, _norm(body)) not in delivered:
                        key = f"{who}: {body}"
                        emit(ts, Op("answer", REPORT, key, value=body, refs=[(REPORT, key)], source=source))
            elif role == "toolResult" and m.get("toolName") == IPYTHON:
                code = pending.pop(m.get("toolCallId", ""), "")
                det = m.get("details") if isinstance(m.get("details"), dict) else {}
                ok = not m.get("isError", False) and det.get("status", "ok") != "error"
                output = _text(m.get("content"))
                for rx in (re.compile(p) for p in FILE_WRITE_PATTERNS):
                    for mm in rx.finditer(code):
                        p = mm.group(1)
                        known_files.add(p)
                        idx = emit(ts, Op("set", "file", p, ok=ok, source=source))
                        if ok:
                            claimed.append((idx, p))
                for rx, kind in writes:
                    for mm in rx.finditer(code):
                        gd = mm.groupdict()
                        emit(ts, Op("set", kind, str(gd["key"]), value=gd.get("value"), ok=ok, source=source))
                handles.extend(_spawn_handles(output, m.get("details")))
            elif role == "custom" and m.get("customType") in RECEIVED:
                for op in resolver.ops(_text(m.get("content")), known_files, _sender(m.get("details"), source)):
                    emit(ts, op)
                who = _delivered_by(m.get("details")) if m.get("customType") == AGENT_MESSAGE else None
                if who is not None:
                    for _, body in message_parts(_text(m.get("content"))) or [(who, _text(m.get("content")).strip())]:
                        emit(ts, Op("set", REPORT, f"{who}: {body}", value=body, source=f"child:{who}"))
        elif t == "custom_message" and e.get("customType") in RECEIVED:
            for op in resolver.ops(_text(e.get("content")), known_files, _sender(e.get("details"), source)):
                emit(ts, op)
            who = _delivered_by(e.get("details")) if e.get("customType") == AGENT_MESSAGE else None
            if who is not None:
                for _, body in message_parts(_text(e.get("content"))) or [(who, _text(e.get("content")).strip())]:
                    emit(ts, Op("set", REPORT, f"{who}: {body}", value=body, source=f"child:{who}"))
        elif t == "compaction":
            for op in resolver.ops(current_state_section(e.get("summary", "")), known_files, "compaction"):
                emit(ts, op)
        elif t == "git_state":
            snap = _git_paths(e.get("git"))
            if prev_git is not None:
                for p, st in snap.items():
                    if p.startswith("\0"):
                        continue
                    if p not in prev_git:
                        emit(ts, Op("add", "file", p, value=p, source=source) if st.upper().startswith(("A", "?"))
                             else Op("set", "file", p, source=source))
                        known_files.add(p)
                    elif prev_git[p] != st:
                        emit(ts, Op("set", "file", p, source=source))
            # a cell-claimed file write the next snapshot never shows is a rejected op
            if isinstance(e.get("git"), dict) and any(k in e["git"] for k in GIT_FILE_FIELDS):
                for idx, p in claimed:
                    if p not in snap:
                        out[idx].op.ok = False
                claimed = []
            prev_git = snap
    return out, handles


def _sender(details: Any, default: str) -> str:
    """`root` for a task from the parent; `child:<name>` for a child's reply or exit notice."""
    if isinstance(details, dict):
        if details.get("fromRelationship") == "parent":
            return "root"
        s = details.get("from") or details.get("sender")
        if isinstance(s, dict):
            name = s.get("sessionName") or s.get("activeSessionId") or s.get("sessionId")
            if name:
                return f"child:{name}"
        name = details.get("senderName") or details.get("sessionName") or details.get("childName")
        if name:
            return f"child:{name}"
    return default


def _spawn_handles(output: str, details: Any) -> List[str]:
    """Child session directories printed by rlm.spawn: RLMSpawnHandle(..., session_dir=PosixPath('...'))."""
    text = output + ("\n" + str(details.get("stdout", "")) if isinstance(details, dict) else "")
    found = re.findall(r"""session_dir=(?:\w*Path\()?['"]([^'"]+)['"]""", text)
    return list(dict.fromkeys(found))


def _is_child_of(path: str, root_path: str, root_id: Optional[str]) -> bool:
    try:
        header, _ = _read_jsonl(path)
    except (OSError, ValueError):
        return False
    if header.get("type") != "session":
        return False
    parent = header.get("parentSession") or ""
    return (os.path.realpath(parent) == os.path.realpath(root_path)
            or (bool(root_id) and os.path.basename(parent) == f"{root_id}.jsonl"))


def _sessions_in(d: str) -> List[str]:
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.endswith(".jsonl") and f != "semantic-edges.jsonl") if os.path.isdir(d) else []


def discover_children(root_path: str, handles: Iterable[str] = ()) -> Dict[str, str]:
    """{child name: session path} for children the root spawned.

    Children live in <sessions>/../session-artifacts/<root id>/sub-*/<child id>.jsonl; the spawn handles in the
    root's cell output name those directories. Both are searched, and a file counts only when its header names
    the root as parentSession.
    """
    root_header, _ = _read_jsonl(root_path)
    root_id = root_header.get("id")
    base = os.path.dirname(os.path.abspath(root_path))
    dirs = list(handles)
    art = os.path.join(os.path.dirname(base), "session-artifacts", str(root_id))
    if os.path.isdir(art):
        dirs += [os.path.join(art, d) for d in sorted(os.listdir(art))]
    dirs.append(base)
    out: Dict[str, str] = {}
    for d in dirs:
        for p in _sessions_in(d):
            if os.path.realpath(p) != os.path.realpath(root_path) and p not in out.values() and _is_child_of(p, root_path, root_id):
                out[_child_name(p)] = p
    return out


def _child_name(path: str) -> str:
    header, entries = _read_jsonl(path)
    names = [e.get("name") for e in entries if e.get("type") == "session_info" and e.get("name")]
    return str(names[-1] if names else header.get("name") or os.path.splitext(os.path.basename(path))[0])


def load_session_timed(path: str, key_pattern: Optional[str] = None, kind: str = "fact",
                       write_patterns: Optional[List[WritePattern]] = None, seed_ops: Optional[List[Op]] = None,
                       children: Optional[Dict[str, str]] = None, qualify: bool = False) -> List[Tuple[str, Op]]:
    """load_session, with each op's ISO timestamp (seed ops carry ""). qualify writes each stated value as
    "<key>=<value>", so a value is matched only against the same fact's history, never another fact's."""
    resolver = _Resolver(key_pattern, kind, qualify)
    writes = [(w.compiled(), w.kind) for w in (write_patterns or [])]
    known: set = set()
    root, handles = _session_ops(path, "root", resolver, writes, known)
    streams = [root]
    kids = children if children is not None else discover_children(path, handles)
    for name, cpath in sorted(kids.items()):
        ops, _ = _session_ops(cpath, f"child:{name}", resolver, writes, known)
        streams.append(ops)
    merged = [t for _, _, _, t in sorted((t.ts, i, t.order, t) for i, s in enumerate(streams) for t in s)]
    out: List[Tuple[str, Op]] = []
    for op in seed_ops or []:
        op.source = op.source or "seed"
        out.append(("", op))
    out.extend((t.ts, t.op) for t in merged)
    for i, (_, op) in enumerate(out):
        op.step = i
    return out


def load_session(path: str, key_pattern: Optional[str] = None, kind: str = "fact",
                 write_patterns: Optional[List[WritePattern]] = None, seed_ops: Optional[List[Op]] = None,
                 children: Optional[Dict[str, str]] = None) -> List[Op]:
    """Merge a root session and its children into one op stream.

    key_pattern: a regex for the names of facts the agents track (e.g. r"e\\d+"); mentions of these in agent
        messages, compaction summaries and child task prompts become answers citing them, with the stated value
        when one is given.
    kind: the op kind for those facts.
    write_patterns: skill calls that commit facts, as regexes with named groups `key` and `value`.
    seed_ops: facts committed before the session began (for instance a starting state given in the prompt).
    children: an explicit {name: session path}; otherwise children are discovered next to the root.
    """
    return [op for _, op in load_session_timed(path, key_pattern, kind, write_patterns, seed_ops, children)]
