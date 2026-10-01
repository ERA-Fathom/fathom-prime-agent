# fathom-prime-agent

Read [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) sessions with the
[fathom read](https://github.com/ERA-Fathom/fathom): find the steps where an orchestrator acts on
something that is not in its committed state, such as a child's message that was never delivered.

It works in two places. In Prime Agent, installed as a package, an extension reads the orchestrator's session at
the end of every turn and records the verdict in the session; it never changes what the agent sees or does. After a
run, `prime-fathom read` reads a saved session from Python, the orchestrator's and its children's.

## Install in Prime Agent

```bash
prime-agent package install git:github.com/ERA-Fathom/fathom-prime-agent
```

The package can also be listed in the `packages` array of Prime Agent's `settings.json`.

```json
{
  "packages": ["git:github.com/ERA-Fathom/fathom-prime-agent"]
}
```

At every turn end the extension posts the orchestrator's session, as fathom ops, to the read and appends the verdict
to the session as a `fathom` entry: `{turn, ops, coherent, findings}`. Children's sessions are not read live; the
orchestrator's is, and child messages appear in it as they are delivered.

Every setting is optional and comes from the environment.

| variable | meaning |
|---|---|
| `FATHOM_API_KEY` | a free key (`fathom key you@example.com`, from `pip install fathom-read`); without one, the anonymous daily limit applies |
| `FATHOM_ENDPOINT` | the read service, default `https://read.embeddedriskanalytics.com` |
| `FATHOM_FACTS` | a regex naming facts your agents track, for example `e\d+`, so stated values are read as claims |
| `FATHOM_KIND` | the op kind for those facts (default `fact`) |
| `FATHOM_WRITES` | JSON `[{"regex": "...(?<key>...)...(?<value>...)", "kind": "..."}]` for skill calls that commit facts |
| `FATHOM_SEED_OPS` | a JSON op list committed before the session began |
| `FATHOM_MODE` | `observe` (default) or `off` |

## Read a saved session

```bash
pip install git+https://github.com/ERA-Fathom/fathom-prime-agent
prime-fathom read ~/.prime/agent/sessions/<session>.jsonl
prime-fathom read <session>.jsonl --json          # the op stream, without calling the read
prime-fathom read <session>.jsonl --facts 'e\d+'  # also read stated values of facts named e0, e1, ...
```

The reader walks the orchestrator's live branch, finds the child sessions it spawned, and merges them into one op
stream in time order. The exit code is 0 when the read finds the session coherent, and 1 otherwise.

The same reader is a Python function, and its op stream goes to the read client.

```python
from prime_fathom import load_session
from fathom_read.client import read

verdict = read(load_session("session.jsonl"))
for f in verdict.findings:
    print(f.step, f.kind, f.key, f.detail)
```

## What the read flags

Each session event becomes an op: a commit to the agent's state, or a claim that cites what is already committed.
The read reports every step that acts on something the committed state does not hold.

| session event | op |
|---|---|
| a message a child delivers | the report `<child>: <message>` is committed |
| an `[agent-message from <child>]` header in the orchestrator's own text, when no delivery from that child carries the same message anywhere in the session | a claim citing that report. **Flagged**: the orchestrator is quoting a message it never received |
| a file an ipython cell writes, or a change in a `git_state` snapshot | the file is committed |
| a skill call matching `FATHOM_WRITES` | the fact is committed |
| a named fact (`FATHOM_FACTS`) in a received message, or in the current-state section of a compaction summary | a claim citing that fact. **Flagged** if the fact was never committed |

A claimed child message is compared with the delivered messages of the whole session. So an orchestrator that quotes
a message a few seconds before its delivery entry is written is not flagged; one that writes a child's message itself,
with content no child delivered, is.

Each finding names the step, its kind, the key it cites, and why it was flagged.

```
incoherent: 1 findings over 9 ops
  step 1 stale_reference r0t1: delta = 99: step 1 acts on report 'r0t1: delta = 99', which is not in the committed state.
```

The read service, its finding kinds and its limits are documented at
[github.com/ERA-Fathom/fathom](https://github.com/ERA-Fathom/fathom).

## Data

The extension and the reader send the read service an op stream. It includes the text of every message a child
delivers and of every child message the orchestrator claims in its own text, file paths, fact names, and stated
values. Full prompts, model responses and all other tool output stay on your machine. Set `FATHOM_MODE=off` to stop
the extension.

## License

MIT


---

If the read caught something in your own run, a star on this repository helps other teams find it.
