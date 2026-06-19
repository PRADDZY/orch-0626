"""CLI entry point for claim prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from claimguard.config import AppConfig
from claimguard.pipeline import ClaimReviewer, read_csv_rows, write_csv_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the claim-review pipeline.")
    parser.add_argument("--input", default="dataset/claims.csv", help="Input CSV path relative to repo root.")
    parser.add_argument("--output", default="output.csv", help="Output CSV path relative to repo root.")
    parser.add_argument(
        "--strategy",
        choices=["hybrid", "retrieval", "text_baseline"],
        default="retrieval",
        help="Prediction strategy to run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_repo(repo_root)
    reviewer = ClaimReviewer(config)
    input_path = repo_root / args.input
    output_path = repo_root / args.output
    rows = read_csv_rows(input_path)
    strategy = "hybrid" if args.strategy == "hybrid" else args.strategy
    if strategy == "hybrid" and not config.enable_live_models:
        strategy = "retrieval"
    predictions = reviewer.predict_rows(rows, strategy=strategy)
    write_csv_rows(output_path, [prediction.values for prediction in predictions])
    print(f"Wrote {len(predictions)} rows to {output_path}")
    print(f"Strategy used: {strategy}")
    print(f"Live models enabled: {config.enable_live_models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
