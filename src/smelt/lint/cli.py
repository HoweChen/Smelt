"""CLI entry: smelt validate <path> (static lint)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smelt.lint.checks import run_checks
from smelt.lint.loader import SkillLoadError, discover_skills, load_skill
from smelt.lint.report import render_json, render_markdown, render_text
from smelt.lint.scorer import build_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smelt validate",
        description="Statically lint the quality of agent skills (SKILL.md)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="lint one skill directory or a root containing many skills")
    v.add_argument("path", type=Path, help="skill directory or its parent directory")
    v.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="output format (default: text)",
    )
    v.add_argument("--output", type=Path, default=None, help="write the report to a file instead of stdout")
    v.add_argument(
        "--fail-under",
        type=float,
        default=60.0,
        help="exit non-zero when the total score is below this (or grade is F); default 60",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "validate":
        try:
            skill_dirs = discover_skills(args.path)
        except SkillLoadError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        reports = []
        for d in skill_dirs:
            doc = load_skill(d)
            reports.append(build_report(doc, run_checks(doc)))

        if args.format == "json":
            out = render_json(reports)
        elif args.format == "markdown":
            out = render_markdown(reports)
        else:
            out = "\n".join(render_text(r) for r in reports)

        if args.output:
            args.output.write_text(out + "\n", encoding="utf-8")
            print(f"report written to {args.output}")
        else:
            print(out)

        ok = all(r.total_score >= args.fail_under and r.passed for r in reports)
        return 0 if ok else 1

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
