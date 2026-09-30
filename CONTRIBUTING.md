# Contributing

Issues and pull requests are welcome.

1. Create a virtual environment with Python 3.11 or later and run `pip install -e ".[dev]"`.
2. Make the change with a test that fails without it.
3. Run `make lint test` (ruff, ruff format, mypy and pytest) before opening a pull request.
4. Use synthetic data only (see `fraud_detection.synthetic`). Never commit real transaction data.

`legacy/` holds the original script for reference; please do not edit it.
