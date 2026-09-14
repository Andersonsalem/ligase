.PHONY: setup test lint format gate

setup:
    uv sync --all-extras
    uv run pre-commit install

test:
    uv run pytest -m "not gpu" --cov=ligase --cov-report=term

lint:
    uv run ruff check . && uv run ruff format --check .

format:
    uv run ruff format . && uv run ruff check --fix .

gate: lint test
