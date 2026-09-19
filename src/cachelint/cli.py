"""``cachelint`` command line: report over a JSONL trace, or lint one request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cachelint import findings as F
from cachelint.analyze import analyze
from cachelint.detectors import lint_request
from cachelint.providers import get_provider
from cachelint.recorder import load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cachelint", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="analyze a recorded JSONL trace")
    rep.add_argument("trace", type=Path)
    rep.add_argument("--json", action="store_true", help="machine-readable output")
    rep.add_argument(
        "--fail-on",
        default="",
        help="comma-separated finding codes that make the exit status non-zero (e.g. CL001,CL002)",
    )

    lint = sub.add_parser("lint", help="static checks on one request body (JSON file or '-')")
    lint.add_argument("request", type=str)
    lint.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    lint.add_argument("--json", action="store_true")

    codes = sub.add_parser("codes", help="list finding codes")

    args = parser.parse_args(argv)
    if args.cmd == "report":
        return _report(args)
    if args.cmd == "lint":
        return _lint(args)
    if args.cmd == "codes":
        for code, desc in sorted(F.DESCRIPTIONS.items()):
            print(f"{code}  {desc}")
        return 0
    codes.print_help()
    return 2


def _report(args: argparse.Namespace) -> int:
    report = analyze(load_jsonl(args.trace))
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.to_text())
    fail_on = {c.strip() for c in args.fail_on.split(",") if c.strip()}
    return 1 if any(f.code in fail_on for f in report.findings) else 0


def _lint(args: argparse.Namespace) -> int:
    raw = sys.stdin.read() if args.request == "-" else Path(args.request).read_text("utf-8")
    body = json.loads(raw)
    findings = lint_request(get_provider(args.provider), body)
    if args.json:
        print(json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2))
    elif not findings:
        print("cachelint: no findings")
    else:
        for f in findings:
            where = f" at {f.path}+{f.offset}" if f.path else ""
            print(f"{f.severity:7} {f.code} {f.message}{where}")
            if f.hint:
                print(f"        → {f.hint}")
    return 1 if any(f.severity == "error" for f in findings) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
