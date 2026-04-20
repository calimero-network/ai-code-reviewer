.PHONY: ci lint typecheck test fix install

# Run the full CI suite locally — mirrors GitHub Actions ci.yaml
ci: lint typecheck test

lint:
	ruff check . --output-format=github
	ruff format --check .

typecheck:
	mypy src/ --ignore-missing-imports

test:
	pytest -v --cov=ai_reviewer

# Auto-fix all lint and format issues
fix:
	ruff check . --fix
	ruff format .

install:
	pip install -e ".[dev]"
