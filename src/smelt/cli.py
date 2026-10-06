"""smelt CLI entry: run case files, evaluate skills, validate SKILL.md statically."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from smelt.case import SmeltCase
from smelt.evaluate import evaluate_skill
from smelt.results import CaseResult


def _load_cases(path: Path) -> list[SmeltCase]:
    """Import a .py file and collect module-level SmeltCase instances / a cases list."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load case file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    found: list[SmeltCase] = []
    explicit = getattr(module, "cases", None)
    if explicit is not None:
        found.extend(c for c in explicit if isinstance(c, SmeltCase))
    else:
        found.extend(v for v in vars(module).values() if isinstance(v, SmeltCase))
    return found


def _cmd_run(paths: list[Path], *, verbose: bool) -> int:
    cases: list[SmeltCase] = []
    for p in paths:
        try:
            cases.extend(_load_cases(p))
        except Exception as e:  # noqa: BLE001
            print(f"✘ failed to load {p}: {e}", file=sys.stderr)
            return 2
    if not cases:
        print("no SmeltCase found (define module-level new_case(...) or cases = [...] in the case file)", file=sys.stderr)
        return 2

    results: list[CaseResult] = [c.run() for c in cases]
    for r in results:
        print(r.summary() if verbose or not r.passed else f"✔ case {r.case_name!r}  score={r.score:.2f}")
    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """smelt evaluate: comprehensively evaluate a skill and generate a report."""
    judge = None
    if args.judge_model:
        try:
            from smelt.given.agents.llm import OpenAIChatClient

            judge = OpenAIChatClient(args.judge_model, base_url=args.base_url)
        except ImportError as e:
            print(f"✘ {e}", file=sys.stderr)
            return 2
    elif args.writing or args.suggestions:
        print("ℹ no --judge-model: writing review and suggestions will be marked as skipped", file=sys.stderr)

    builder = evaluate_skill(args.path, judge=judge)
    if args.cases:
        cases: list[SmeltCase] = []
        for p in args.cases:
            try:
                cases.extend(_load_cases(p))
            except Exception as e:  # noqa: BLE001
                print(f"✘ failed to load cases {p}: {e}", file=sys.stderr)
                return 2
        builder = builder.with_cases(*cases)
    if not args.lint:
        builder = builder.with_lint(False)
    if not args.writing:
        builder = builder.with_writing(enabled=False)
    if not args.suggestions:
        builder = builder.with_suggestions(enabled=False)

    try:
        evaluation = builder.run()
    except Exception as e:  # noqa: BLE001
        print(f"✘ evaluation failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    out = evaluation.save(args.output) if args.output else None
    if out:
        print(f"report written to {out}")
    else:
        print(evaluation.to_markdown())
    score = evaluation.overall_score
    return 0 if score is not None and score >= args.fail_under else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smelt", description="Smelt — a behavior-verification framework for agent skills")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute case files (.py with module-level SmeltCase)")
    run.add_argument("files", type=Path, nargs="+", help="case file paths")
    run.add_argument("-v", "--verbose", action="store_true", help="always print per-expectation details")

    validate = sub.add_parser("validate", help="statically lint SKILL.md (former skillcheck capability)")
    validate.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to smelt validate")

    ev = sub.add_parser("evaluate", help="evaluate a skill: behavior + writing + lint scores + suggestions")
    ev.add_argument("path", type=Path, help="skill directory or SKILL.md path")
    ev.add_argument("--cases", type=Path, nargs="*", default=None, help="behavior case files (.py)")
    ev.add_argument("--judge-model", default=None, help="judge model name (OpenAI-compatible API)")
    ev.add_argument("--base-url", default=None, help="base_url for the judge model")
    ev.add_argument("--output", type=Path, default=None, help="report output path (.json → JSON, otherwise Markdown)")
    ev.add_argument("--fail-under", type=float, default=60.0, help="exit non-zero below this score (default 60)")
    ev.add_argument("--no-lint", dest="lint", action="store_false", help="disable the static lint part")
    ev.add_argument("--no-writing", dest="writing", action="store_false", help="disable the LLM writing review part")
    ev.add_argument("--no-suggestions", dest="suggestions", action="store_false", help="disable suggestion generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.files, verbose=args.verbose)
    if args.command == "validate":
        from smelt.lint.cli import main as lint_main

        return lint_main(["validate", *(args.args or ["--help"])])
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
