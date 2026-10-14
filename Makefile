.PHONY: run demo data test lint format fix typecheck check

run:
	uv run itg-transformers

demo:
	uv run python app.py

data:
	uv run python -c "from itg_transformers.data import download_dataset; download_dataset()"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

check:
	@./scripts/check.sh
