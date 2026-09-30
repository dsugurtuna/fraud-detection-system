"""Command-line interface: ``python -m fraud_detection <command>``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .model import TrainConfig
from .pipeline import PipelineResult, load_transactions, run_pipeline
from .synthetic import make_transactions


def _print(result: PipelineResult) -> None:
    m = result.test_metrics
    print(json.dumps({"rows": result.rows, "frauds": result.frauds}))
    print(f"selected model:            {result.selected_model}")
    print(f"test ROC AUC (classifier): {m.roc_auc:.3f}")
    print(f"test PR AUC (classifier):  {m.pr_auc:.3f}")
    print(f"months in test:            {m.months_in_test}")
    print(f"reviews per month:         {m.n_reviews_per_month}")
    print(f"fraud value captured:      {m.captured_value_gbp:,.2f}")
    print(f"random-review baseline:    {m.baseline_random_value_gbp:,.2f}")
    print(f"uplift vs random:          {m.uplift_vs_random_x:.2f}x")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraud_detection")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run end to end on synthetic data")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--rows", type=int, default=30_000)
    demo.add_argument("--capacity", type=int, default=50)
    demo.add_argument("--iterations", type=int, default=300)

    run = sub.add_parser("run", help="run on transactions + labels CSV files")
    run.add_argument("--transactions", required=True)
    run.add_argument("--labels", required=True)
    run.add_argument("--val-start", required=True)
    run.add_argument("--test-start", required=True)
    run.add_argument("--capacity", type=int, default=400)
    run.add_argument("--iterations", type=int, default=1000)

    args = parser.parse_args(argv)
    config = TrainConfig(iterations=args.iterations)
    if args.command == "demo":
        df = make_transactions(n=args.rows, seed=args.seed)
        result = run_pipeline(
            df, "2024-06-01", "2024-07-15", capacity=args.capacity, config=config
        )
    else:
        df = load_transactions(args.transactions, args.labels)
        result = run_pipeline(
            df, args.val_start, args.test_start, capacity=args.capacity, config=config
        )
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
