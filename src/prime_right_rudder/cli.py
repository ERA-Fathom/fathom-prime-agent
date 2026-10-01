"""prime-right-rudder read <session.jsonl> [--json] [--key KEY]"""
from __future__ import annotations

import argparse
import json
import sys

from .reader import load_session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prime-right-rudder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="read a Prime Agent session and its children")
    r.add_argument("session")
    r.add_argument("--json", action="store_true", help="print the op stream instead of calling the read")
    r.add_argument("--key", help="a right-rudder key (free: POST /v1/keys); default is the anonymous limit")
    r.add_argument("--facts", help="regex naming the facts the agents track, e.g. 'e\\d+'")
    r.add_argument("--kind", default="fact")
    a = ap.parse_args(argv)
    ops = load_session(a.session, key_pattern=a.facts, kind=a.kind)
    if a.json:
        json.dump([o.as_dict() for o in ops], sys.stdout, indent=1)
        print()
        return 0
    from right_rudder.client import read
    v = read(ops, key=a.key)
    print(f"{'coherent' if v.coherent else 'incoherent'}: {len(v.findings)} findings over {v.ops_read} ops")
    for f in v.findings:
        print(f"  step {f.step} {f.kind} {f.key}: {f.detail}")
    return 0 if v.coherent else 1


if __name__ == "__main__":
    sys.exit(main())
