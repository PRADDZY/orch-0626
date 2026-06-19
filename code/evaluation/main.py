"""Evaluation entry point for sample-set comparisons and report generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claimguard.config import AppConfig
from claimguard.pipeline import (
    ClaimReviewer,
    build_operational_notes,
    evaluate_predictions,
    read_csv_rows,
    write_csv_rows,
)
from claimguard.reporting import write_html_report, write_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate claim-review strategies on sample data.")
    parser.add_argument("--sample", default="dataset/sample_claims.csv", help="Sample CSV path relative to repo root.")
    parser.add_argument(
        "--report",
        default="code/evaluation/evaluation_report.md",
        help="Markdown report path relative to repo root.",
    )
    parser.add_argument(
        "--html",
        default="code/evaluation/report/index.html",
        help="HTML explorer path relative to repo root.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = AppConfig.from_repo(repo_root)
    reviewer = ClaimReviewer(config)
    sample_rows = read_csv_rows(repo_root / args.sample)

    baseline_predictions = reviewer.predict_rows(sample_rows, strategy="text_baseline")
    retrieval_predictions = reviewer.predict_rows(sample_rows, strategy="retrieval")
    hybrid_predictions = reviewer.predict_rows(sample_rows, strategy="hybrid") if config.enable_live_models else retrieval_predictions

    baseline_metric = evaluate_predictions(
        sample_rows,
        [prediction.values for prediction in baseline_predictions],
        strategy_name="text_baseline",
        notes="Transcript parsing with image quality checks only.",
    )
    retrieval_metric = evaluate_predictions(
        sample_rows,
        [prediction.values for prediction in retrieval_predictions],
        strategy_name="retrieval",
        notes="Offline retrieval plus rule arbitration.",
    )
    hybrid_metric = evaluate_predictions(
        sample_rows,
        [prediction.values for prediction in hybrid_predictions],
        strategy_name="hybrid" if config.enable_live_models else "retrieval",
        notes="Live per-image multimodal review plus claim aggregation when keys are present, otherwise retrieval plus rule arbitration.",
    )
    candidates = [retrieval_metric]
    if config.enable_live_models:
        candidates.append(hybrid_metric)
    final_metric = max(candidates, key=lambda metric: metric.exact_match_accuracy)
    final_predictions = hybrid_predictions if final_metric.name == "hybrid" else retrieval_predictions

    avg_images = sum(len(row["image_paths"].split(";")) for row in sample_rows) / max(1, len(sample_rows))
    ops = build_operational_notes(reviewer, total_rows=len(sample_rows), avg_images_per_row=avg_images)
    metrics = [baseline_metric, retrieval_metric]
    if config.enable_live_models:
        metrics.append(hybrid_metric)
    write_markdown_report(repo_root / args.report, metrics, final_metric.name, ops)
    write_html_report(
        repo_root / args.html,
        repo_root=repo_root,
        expected_rows=sample_rows,
        predicted_rows=[prediction.values for prediction in final_predictions],
    )
    write_csv_rows(
        repo_root / "code" / "evaluation" / f"{final_metric.name}_sample_predictions.csv",
        [prediction.values for prediction in final_predictions],
    )
    if config.enable_live_models:
        write_csv_rows(
            repo_root / "code" / "evaluation" / "hybrid_sample_predictions.csv",
            [prediction.values for prediction in hybrid_predictions],
        )
    print(f"Report written to {repo_root / args.report}")
    print(f"HTML explorer written to {repo_root / args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
