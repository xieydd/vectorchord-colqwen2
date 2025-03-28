PY_SOURCE=.

lint:
	@uv run -- ruff check ${PY_SOURCE}

typecheck:
	@uv run -- mypy --non-interactive --install-types ${PY_SOURCE}

format:
	@uv run -- ruff check --fix ${PY_SOURCE}
	@uv run -- ruff format ${PY_SOURCE}
	
.PHONY: lint format