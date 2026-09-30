.PHONY: install dev test lint format clean docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .
	mypy

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info

docker:
	docker build -t fraud-detection-system .
