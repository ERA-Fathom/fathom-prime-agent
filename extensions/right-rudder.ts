/**
 * Right Rudder for Prime Agent: read the orchestrator's session as it runs.
 *
 * At each turn_end the extension maps the orchestrator's live branch to Right Rudder ops, calls the right-rudder read
 * (POST /v1/read), and records the verdict in the session with pi.appendEntry("fathom", ...). It never changes
 * what the agent sees or does.
 *
 * What it maps (the same mapping as the Python reader, `prime-right-rudder read`, over the orchestrator's session):
 *   - a file an ipython cell writes                  -> set  file <path>
 *   - a skill call matching RIGHT_RUDDER_WRITES            -> set  <kind> <key>
 *   - a git_state change                             -> add / set file <path>
 *   - a message a child delivers                     -> set  report "<child>: <message>"
 *   - an "[agent-message from <child>]" header in the orchestrator's own text, with no delivery from that child
 *     carrying the same message anywhere in the session
 *                                                    -> answer citing that report: the read flags it, because
 *                                                       the text claims a message that was never delivered
 *   - named facts (RIGHT_RUDDER_FACTS) in received messages and in a compaction summary's current-state section
 *                                                    -> answer ops citing them
 *
 * Configuration (environment, all optional; the FATHOM_ names still read as fallbacks):
 *   RIGHT_RUDDER_MODE      observe (default) | off
 *   RIGHT_RUDDER_ENDPOINT  service base URL, default https://read.embeddedriskanalytics.com
 *   RIGHT_RUDDER_API_KEY   a free key from POST /v1/keys; without one the anonymous limit applies
 *   RIGHT_RUDDER_FACTS     regex naming the facts your agents track, e.g. e\d+
 *   RIGHT_RUDDER_KIND      op kind for those facts (default "fact")
 *   RIGHT_RUDDER_WRITES    JSON [{"regex": "...(?<key>...)...(?<value>...)", "kind": "..."}] for skill calls that commit facts
 *   RIGHT_RUDDER_SEED_OPS  path to a JSON op list committed before the session began
 */
import { existsSync, readFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export type Ref = [string, string];
export interface Op {
	op: "set" | "add" | "remove" | "rename" | "answer" | "commit";
	kind: string;
	key: string;
	value?: string | null;
	ok?: boolean;
	refs?: Ref[];
	step?: number;
	source?: string;
}

interface Entry {
	type: string;
	id: string;
	parentId: string | null;
	timestamp?: string;
	[k: string]: any;
}

export interface MapConfig {
	facts?: RegExp;
	kind: string;
	writes: { re: RegExp; kind: string }[];
}

const RECEIVED = new Set(["agent_message", "rlm_child_terminal_notice"]);
const REPORT = "report";
const HEADER = /\[agent-message from (?:child:)?([^\]\s]+)\]/g;
const GIT_FILE_FIELDS = ["files", "changedFiles", "changes", "status", "dirtyFiles", "entries"];
const FILE_WRITES = [
	/open\(\s*[rbfu]?['"]([^'"]+)['"]\s*,\s*[rbfu]?['"][wax]/g,
	/Path\(\s*[rbfu]?['"]([^'"]+)['"]\s*\)\s*\.write_(?:text|bytes)\(/g,
	/(?:write_file|edit_file|replace_in_file|apply_edit)\(\s*[rbfu]?['"]([^'"]+)['"]/g,
];
const FILE_REF = /(?<![\w/.-])((?:[\w.-]+\/)*[\w.-]+\.[A-Za-z0-9]{1,8})(?![\w/-])/g;
const CURRENT_LABEL = /^(\s*)(#+\s*|[-*]\s+)?\**\s*current\b[^\n]{0,40}?\b(?:state|values?)\b/i;

export function configFromEnv(env = process.env): MapConfig {
	const writes = (env.RIGHT_RUDDER_WRITES ?? env.FATHOM_WRITES) ? (JSON.parse((env.RIGHT_RUDDER_WRITES ?? env.FATHOM_WRITES)) as { regex: string; kind: string }[]) : [];
	return {
		facts: (env.RIGHT_RUDDER_FACTS ?? env.FATHOM_FACTS) ? new RegExp((env.RIGHT_RUDDER_FACTS ?? env.FATHOM_FACTS)) : undefined,
		kind: (env.RIGHT_RUDDER_KIND ?? env.FATHOM_KIND) || "fact",
		writes: writes.map((w) => ({ re: new RegExp(w.regex, "g"), kind: w.kind })),
	};
}

function text(content: any): string {
	if (content == null) return "";
	if (typeof content === "string") return content;
	return (content as any[])
		.filter((p) => p && p.type === "text")
		.map((p) => p.text ?? "")
		.join("\n");
}

function cellCode(args: any): string {
	if (typeof args === "string") return args;
	if (args && typeof args === "object") {
		for (const k of ["code", "source", "cell", "input"]) if (typeof args[k] === "string") return args[k];
		for (const v of Object.values(args)) if (typeof v === "string") return v;
	}
	return "";
}

const norm = (t: string) => (t ?? "").split(/\s+/).filter(Boolean).join(" ").toLowerCase();

/** [sender, message] for every "[agent-message from <sender>]" header in a text: the first paragraph after it. */
export function messageParts(t: string): [string, string][] {
	const out: [string, string][] = [];
	for (const m of (t ?? "").matchAll(HEADER)) {
		const rest = t.slice((m.index ?? 0) + m[0].length).replace(/^\s+/, "");
		const next = rest.search(/\[agent-message from (?:child:)?[^\]\s]+\]/);
		let body = (next >= 0 ? rest.slice(0, next) : rest).trim();
		body = body.split(/\n\s*\n/)[0].trim();
		if (body) out.push([m[1], body]);
	}
	return out;
}

/** The child that sent a delivered message; undefined when it came from the parent. */
function deliveredBy(details: any): string | undefined {
	if (details?.fromRelationship !== "child") return undefined;
	const s = details?.from ?? details?.sender;
	if (s?.sessionName) return String(s.sessionName);
	const n = details?.senderName ?? details?.childName;
	return n ? String(n) : undefined;
}

function sender(details: any, dflt: string): string {
	if (details?.fromRelationship === "parent") return "root";
	const s = details?.from ?? details?.sender;
	const name = (s && (s.sessionName ?? s.activeSessionId ?? s.sessionId)) ?? details?.senderName ?? details?.sessionName ?? details?.childName;
	return name ? `child:${name}` : dflt;
}

/** The parts of a compaction summary it presents as the current state (history is not a claim). */
export function currentStateSection(summary: string): string {
	const lines = (summary ?? "").split("\n");
	const out: string[] = [];
	let i = 0;
	while (i < lines.length) {
		const m = lines[i].match(CURRENT_LABEL);
		if (!m) {
			i += 1;
			continue;
		}
		const indent = m[1].length;
		const mark = (m[2] ?? "").trim();
		const level = mark.startsWith("#") ? mark.length : null;
		out.push(lines[i].includes(":") ? lines[i].split(":").slice(1).join(":") : "");
		let j = i + 1;
		while (j < lines.length) {
			const line = lines[j];
			if (level !== null) {
				const h = line.match(/^\s*(#+)\s/);
				if (h && h[1].length <= level) break;
			} else if (!line.trim() || line.length - line.trimStart().length <= indent) {
				break;
			}
			out.push(line);
			j += 1;
		}
		i = j;
	}
	return out.join("\n");
}

/** Named facts (with stated values) and known files a text mentions, as answer ops. */
export function mentions(t: string, cfg: MapConfig, known: Set<string>, source: string): Op[] {
	const out: Op[] = [];
	const seen = new Set<string>();
	if (cfg.facts) {
		const keyRe = new RegExp(`^(?:${cfg.facts.source})$`);
		const fact = new RegExp(`(?<![\\w.])['"]?(${cfg.facts.source})['"]?\\s*(?:=|:|is|->|→)\\s*(-?\\d+(?:\\.\\d+)?|[\\w./-]+)`, "g");
		const named = new Set<string>();
		for (const m of t.matchAll(fact)) {
			const key = m[1];
			const val = m[m.length - 1];
			if (seen.has(`${key}\0${val}`) || keyRe.test(val)) continue;
			seen.add(`${key}\0${val}`);
			named.add(key);
			out.push({ op: "answer", kind: cfg.kind, key, value: val, refs: [[cfg.kind, key]], source });
		}
		for (const m of t.matchAll(new RegExp(cfg.facts.source, "g"))) {
			if (named.has(m[0])) continue;
			named.add(m[0]);
			out.push({ op: "answer", kind: cfg.kind, key: m[0], value: null, refs: [[cfg.kind, m[0]]], source });
		}
	}
	for (const m of t.matchAll(FILE_REF)) {
		const p = m[1];
		if ((known.has(p) || p.includes("/")) && !seen.has(`file\0${p}`)) {
			seen.add(`file\0${p}`);
			out.push({ op: "answer", kind: "file", key: p, value: null, refs: [["file", p]], source });
		}
	}
	return out;
}

function gitPaths(git: any): Map<string, string> {
	const out = new Map<string, string>();
	if (!git || typeof git !== "object") return out;
	for (const f of GIT_FILE_FIELDS) {
		const v = git[f];
		if (Array.isArray(v)) {
			for (const x of v) {
				if (typeof x === "string") out.set(x, "M");
				else if (x && typeof x === "object") {
					const p = x.path ?? x.file ?? x.name;
					if (p) out.set(String(p), String(x.status ?? x.state ?? x.change ?? "M"));
				}
			}
		} else if (v && typeof v === "object") {
			for (const [p, s] of Object.entries(v)) out.set(String(p), String(s));
		}
	}
	for (const n of ["head", "commit", "sha", "branch"]) if (typeof git[n] === "string" && !out.has(`\0${n}`)) out.set(`\0${n}`, git[n]);
	return out;
}

/** The delivered messages on a branch, as "<sender>\0<normalised message>". */
export function deliveries(branch: Entry[]): Set<string> {
	const out = new Set<string>();
	for (const e of branch) {
		let content: any;
		let details: any;
		if (e.type === "custom_message" && e.customType === "agent_message") [content, details] = [e.content, e.details];
		else if (e.type === "message" && e.message?.customType === "agent_message") [content, details] = [e.message.content, e.message.details];
		const who = deliveredBy(details);
		if (!who) continue;
		const parts = messageParts(text(content));
		for (const [, body] of parts.length ? parts : [[who, text(content).trim()] as [string, string]]) out.add(`${who}\0${norm(body)}`);
	}
	return out;
}

/** The orchestrator's live branch as ops, in order. Stateless: the whole branch is mapped each time, so a claimed
 *  message whose delivery lands later stops being flagged once it does. */
export function mapBranch(branch: Entry[], cfg: MapConfig, seed: Op[] = []): Op[] {
	const out: Op[] = seed.map((o) => ({ ...o, source: o.source ?? "seed" }));
	const delivered = deliveries(branch);
	const pending = new Map<string, string>();
	const known = new Set<string>();
	let prevGit: Map<string, string> | null = null;
	const received = (content: any, details: any, customType: string) => {
		out.push(...mentions(text(content), cfg, known, sender(details, "root")));
		const who = customType === "agent_message" ? deliveredBy(details) : undefined;
		if (!who) return;
		const parts = messageParts(text(content));
		for (const [, body] of parts.length ? parts : [[who, text(content).trim()] as [string, string]]) {
			out.push({ op: "set", kind: REPORT, key: `${who}: ${body}`, value: body, source: `child:${who}` });
		}
	};
	for (const e of branch) {
		if (e.type === "message") {
			const m = e.message ?? {};
			if (m.role === "assistant") {
				for (const part of m.content ?? []) {
					if (part?.type === "toolCall" && part.name === "ipython") pending.set(part.id, cellCode(part.arguments));
				}
				for (const [who, body] of messageParts(text(m.content))) {
					if (delivered.has(`${who}\0${norm(body)}`)) continue;
					const key = `${who}: ${body}`;
					out.push({ op: "answer", kind: REPORT, key, value: body, refs: [[REPORT, key]], source: "root" });
				}
			} else if (m.role === "toolResult" && m.toolName === "ipython") {
				const code = pending.get(m.toolCallId) ?? "";
				pending.delete(m.toolCallId);
				const ok = !m.isError && (m.details?.status ?? "ok") !== "error";
				for (const re of FILE_WRITES) {
					for (const mm of code.matchAll(new RegExp(re.source, "g"))) {
						known.add(mm[1]);
						out.push({ op: "set", kind: "file", key: mm[1], ok, source: "root" });
					}
				}
				for (const w of cfg.writes) {
					for (const mm of code.matchAll(new RegExp(w.re.source, "g"))) {
						const g = mm.groups ?? {};
						if (g.key) out.push({ op: "set", kind: w.kind, key: g.key, value: g.value ?? null, ok, source: "root" });
					}
				}
			} else if (m.role === "custom" && RECEIVED.has(m.customType)) {
				received(m.content, m.details, m.customType);
			}
		} else if (e.type === "custom_message" && RECEIVED.has(e.customType)) {
			received(e.content, e.details, e.customType);
		} else if (e.type === "compaction") {
			out.push(...mentions(currentStateSection(e.summary ?? ""), cfg, known, "compaction"));
		} else if (e.type === "git_state") {
			const snap = gitPaths(e.git);
			if (prevGit) {
				for (const [p, st] of snap) {
					if (p.startsWith("\0")) continue;
					if (!prevGit.has(p)) {
						out.push(/^[A?]/i.test(st) ? { op: "add", kind: "file", key: p, value: p, source: "root" } : { op: "set", kind: "file", key: p, source: "root" });
						known.add(p);
					} else if (prevGit.get(p) !== st) {
						out.push({ op: "set", kind: "file", key: p, source: "root" });
					}
				}
			}
			prevGit = snap;
		}
	}
	return out.map((o, i) => ({ ...o, step: i }));
}

export function liveBranch(entries: Entry[]): Entry[] {
	if (!entries.length) return [];
	const byId = new Map(entries.map((e) => [e.id, e]));
	const path: Entry[] = [];
	const seen = new Set<string>();
	let cur: Entry | undefined = entries[entries.length - 1];
	while (cur && !seen.has(cur.id)) {
		seen.add(cur.id);
		path.push(cur);
		cur = cur.parentId ? byId.get(cur.parentId) : undefined;
	}
	return path.reverse();
}

export async function read(ops: Op[], env = process.env): Promise<any> {
	const base = ((env.RIGHT_RUDDER_ENDPOINT ?? env.FATHOM_ENDPOINT) || "https://read.embeddedriskanalytics.com").replace(/\/+$/, "");
	// without a key the read runs at the anonymous ("demo") limit
	const headers: Record<string, string> = { "Content-Type": "application/json", Authorization: `Bearer ${(env.RIGHT_RUDDER_API_KEY ?? env.FATHOM_API_KEY) || "demo"}` };
	const r = await fetch(`${base}/v1/read`, { method: "POST", headers, body: JSON.stringify({ ops, supersede: [] }) });
	const out = await r.json().catch(() => ({}));
	if (!r.ok) throw new Error(`/v1/read ${r.status}: ${JSON.stringify(out).slice(0, 200)}`);
	return out;
}

export function seedOps(env = process.env): Op[] {
	const p = (env.RIGHT_RUDDER_SEED_OPS ?? env.FATHOM_SEED_OPS);
	return p && existsSync(p) ? (JSON.parse(readFileSync(p, "utf8")) as Op[]) : [];
}

export function isChildSession(ctx: any): boolean {
	const h = ctx?.sessionManager?.getHeader?.();
	return Boolean(h?.parentSession) || Number(h?.rlmDepth ?? 0) > 0;
}

export default function rightRudder(pi: ExtensionAPI) {
	if ((((process.env.RIGHT_RUDDER_MODE ?? process.env.FATHOM_MODE) ?? process.env.FATHOM_MODE) || "observe").toLowerCase() === "off") return;
	const cfg = configFromEnv();
	const seed = seedOps();
	let turn = 0;
	pi.on("turn_end", async (_event, ctx) => {
		// Prime Agent loads extensions in every session; only the orchestrator's instance reads
		if (isChildSession(ctx)) return;
		turn += 1;
		const ops = mapBranch(liveBranch(ctx.sessionManager.getEntries() as Entry[]), cfg, seed);
		try {
			const verdict = await read(ops);
			pi.appendEntry("fathom", { turn, ops: ops.length, coherent: verdict.coherent, findings: verdict.findings });
		} catch (err) {
			pi.appendEntry("fathom", { turn, error: String(err) });
		}
	});
}
