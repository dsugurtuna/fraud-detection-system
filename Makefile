.PHONY: install dev test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info __pycache__ .pytest_cache
