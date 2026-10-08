"""smelt CLI entry: run case files, evaluate skills, validate SKILL.md statically."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

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
    from smelt.env import _auto_load

    _auto_load()
    judge = None
    judge_model = args.judge_model or os.environ.get("SMELT_JUDGE_MODEL")
    if judge_model:
        try:
            from smelt.given.agents.llm import OpenAIChatClient

            judge = OpenAIChatClient(judge_model, base_url=args.base_url)
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


def _cmd_compare(args: argparse.Namespace) -> int:
    """smelt compare: diff two evaluation reports; exit 1 on regression."""
    from smelt.compare import compare

    try:
        result = compare(args.baseline, args.candidate, min_delta=args.min_delta)
    except Exception as e:  # noqa: BLE001
        print(f"✘ compare failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = result.save(args.output) if args.output else None
    if out:
        print(f"comparison written to {out}")
        print(f"{len(result.improvements)} improved · {len(result.regressions)} regressed")
    else:
        print(result.to_markdown())
    return 1 if result.has_regression else 0


def _role_client(args: argparse.Namespace, role: str) -> Any:
    """Resolve a role's LLM client: CLI flags > SMELT_<ROLE>_*> legacy shared keys."""
    from smelt.llm_config import LLMConfig

    return LLMConfig.from_role(
        role,
        provider=getattr(args, f"{role}_provider", None),
        base_url=getattr(args, f"{role}_base_url", None),
        api_key=getattr(args, f"{role}_api_key", None),
        model=getattr(args, f"{role}_model", None),
    ).build()


def _cmd_doctor(args: argparse.Namespace, *, doctor_llm: Any = None) -> int:
    """smelt doctor: health-check a case suite and the judge."""
    from smelt.challenge.doctor import doctor
    from smelt.llm_config import LLMConfigError

    if doctor_llm is None:
        try:
            doctor_llm = _role_client(args, "doctor")
        except LLMConfigError as e:
            print(f"✘ doctor agent not configured: {e}", file=sys.stderr)
            print("  set SMELT_DOCTOR_MODEL (+ SMELT_DOCTOR_API_KEY / _BASE_URL / _PROVIDER as needed), "
                  "or pass --doctor-model", file=sys.stderr)
            return 2
    try:
        report = doctor(args.cases, skill=args.skill, doctor=doctor_llm, min_score=args.min_score)
    except Exception as e:  # noqa: BLE001
        print(f"✘ doctor failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = report.save(args.output) if args.output else None
    if out:
        print(f"doctor report written to {out}")
    print(report.to_markdown())
    return 0 if report.ok else 1


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
    ev.add_argument("--judge-model", default=None, help="judge model name (OpenAI-compatible API); falls back to SMELT_JUDGE_MODEL")
    ev.add_argument("--base-url", default=None, help="base_url for the judge model")
    ev.add_argument("--output", type=Path, default=None, help="report output path (.json → JSON, otherwise Markdown)")
    ev.add_argument("--fail-under", type=float, default=60.0, help="exit non-zero below this score (default 60)")
    ev.add_argument("--no-lint", dest="lint", action="store_false", help="disable the static lint part")
    ev.add_argument("--no-writing", dest="writing", action="store_false", help="disable the LLM writing review part")
    ev.add_argument("--no-suggestions", dest="suggestions", action="store_false", help="disable suggestion generation")

    cmp_parser = sub.add_parser("compare", help="diff two evaluation reports (.json); exit 1 on regression")
    cmp_parser.add_argument("baseline", type=Path, help="baseline report (.json from evaluate --output)")
    cmp_parser.add_argument("candidate", type=Path, help="candidate report (.json)")
    cmp_parser.add_argument("--min-delta", type=float, default=0.05, help="minimum |Δ| that counts as a change (default 0.05)")
    cmp_parser.add_argument("--output", type=Path, default=None, help="write the comparison report to a file (.json → JSON)")

    doc = sub.add_parser("doctor", help="health-check a case suite (mutation) and the judge (canary)")
    doc.add_argument("cases", type=Path, nargs="+", help="case file paths (.py)")
    doc.add_argument("--skill", type=Path, required=True, help="skill directory or SKILL.md path")
    doc.add_argument("--doctor-provider", default=None, help="openai | anthropic")
    doc.add_argument("--doctor-model", default=None, help="doctor agent model (falls back to SMELT_DOCTOR_MODEL)")
    doc.add_argument("--doctor-base-url", default=None)
    doc.add_argument("--doctor-api-key", default=None)
    doc.add_argument("--min-score", type=float, default=0.8, help="mutation score gate (default 0.8)")
    doc.add_argument("--output", type=Path, default=None, help="report output path (.json → JSON)")
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
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
