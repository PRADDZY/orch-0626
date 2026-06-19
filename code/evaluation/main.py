"""Evaluation entry point for sample-set comparisons and report generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claimguard.config import AppConfig
from claimguard.models import RuntimeStats
from claimguard.pipeline import (
    ClaimReviewer,
    build_operational_notes,
    evaluate_predictions,
    read_csv_rows,
    write_csv_rows,
)
from claimguard.reporting import write_html_report, write_markdown_report


def merge_runtime_stats(reviewers: list[ClaimReviewer]) -> RuntimeStats:
    merged = RuntimeStats()
    for reviewer in reviewers:
        merged.merge(reviewer.runtime_stats)
    return merged


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
    sample_rows = read_csv_rows(repo_root / args.sample)

    baseline_reviewer = ClaimReviewer(config)
    baseline_predictions = baseline_reviewer.predict_rows(sample_rows, strategy="text_baseline")
    retrieval_reviewer = ClaimReviewer(config)
    retrieval_predictions = retrieval_reviewer.predict_rows(sample_rows, strategy="retrieval")
    if config.enable_live_models:
        hybrid_reviewer = ClaimReviewer(config)
        hybrid_predictions = hybrid_reviewer.predict_rows(sample_rows, strategy="hybrid")
        ensemble_reviewer = ClaimReviewer(config)
        ensemble_predictions = ensemble_reviewer.predict_rows(sample_rows, strategy="ensemble")
    else:
        hybrid_reviewer = retrieval_reviewer
        ensemble_reviewer = retrieval_reviewer
        hybrid_predictions = retrieval_predictions
        ensemble_predictions = retrieval_predictions

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
    ensemble_metric = evaluate_predictions(
        sample_rows,
        [prediction.values for prediction in ensemble_predictions],
        strategy_name="ensemble" if config.enable_live_models else "retrieval",
        notes="Retrieval-first arbitration that promotes live multimodal outputs only when the live decision is cleaner or more grounded than the fallback.",
    )
    candidates = [retrieval_metric]
    if config.enable_live_models:
        candidates.extend([hybrid_metric, ensemble_metric])
    final_metric = max(candidates, key=lambda metric: metric.exact_match_accuracy)
    final_predictions_map = {
        "retrieval": retrieval_predictions,
        "hybrid": hybrid_predictions,
        "ensemble": ensemble_predictions,
    }
    final_reviewers = {
        "retrieval": retrieval_reviewer,
        "hybrid": hybrid_reviewer,
        "ensemble": ensemble_reviewer,
    }
    final_predictions = final_predictions_map[final_metric.name]
    if config.enable_live_models:
        evaluation_live_stats = merge_runtime_stats([hybrid_reviewer, ensemble_reviewer])
    else:
        evaluation_live_stats = merge_runtime_stats([retrieval_reviewer])

    avg_images = sum(len(row["image_paths"].split(";")) for row in sample_rows) / max(1, len(sample_rows))
    ops = build_operational_notes(
        final_reviewers[final_metric.name].runtime_stats,
        total_rows=len(sample_rows),
        avg_images_per_row=avg_images,
        strategy_name=final_metric.name,
        live_models_enabled=config.enable_live_models,
        evaluation_stats=evaluation_live_stats,
    )
    metrics = [baseline_metric, retrieval_metric]
    if config.enable_live_models:
        metrics.extend([hybrid_metric, ensemble_metric])
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
        write_csv_rows(
            repo_root / "code" / "evaluation" / "ensemble_sample_predictions.csv",
            [prediction.values for prediction in ensemble_predictions],
        )
    print(f"Report written to {repo_root / args.report}")
    print(f"HTML explorer written to {repo_root / args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
