.PHONY: help dev test coverage demo lint

help:
	@echo "Smelt — a behavior-verification framework for agent skills"
	@echo ""
	@echo "  make dev       install dev dependencies (uv sync --extra dev)"
	@echo "  make test      run the full test suite"
	@echo "  make coverage  run tests with a coverage report (90% gate)"
	@echo "  make demo      run example cases + statically lint examples"
	@echo "  make lint      ruff check"

dev:
	uv sync --extra dev

test:
	uv run pytest -q

coverage:
	rm -f .coverage*
	uv run coverage run -m pytest -q
	uv run coverage report -m

demo:
	uv run smelt run examples/cases/demo_cases.py -v
	uv run smelt validate examples || true

lint:
	uv run ruff check src tests examples/cases
