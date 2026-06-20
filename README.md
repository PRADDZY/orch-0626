# ClaimGuard Ensemble

ClaimGuard Ensemble is a multimodal claim-review pipeline for the HackerRank Orchestrate June 2026 `multi-modal-review` challenge. It reviews claim conversations, local evidence images, user history, and minimum evidence rules to decide whether a claim is supported, contradicted, or not actionable from the submitted evidence.

The final submission strategy is a retrieval-first ensemble:

- a deterministic offline retrieval fallback provides a strong floor
- a live multimodal path can normalize claims and review images
- the ensemble promotes live outputs only when they are cleaner or better grounded than retrieval

On the organizer sample set included in this repo, the latest checked-in fresh evaluation selects `ensemble` as the best strategy with `60.00%` exact row accuracy versus `50.00%` for pure retrieval.

## Solution Summary

- `code/main.py` runs the final pipeline and writes `output.csv`
- `code/evaluation/main.py` benchmarks `text_baseline`, `retrieval`, `hybrid`, and `ensemble`
- `code/evaluation/evaluation_report.md` contains the current sample-set results
- `code/evaluation/report/index.html` is a static explorer for expected vs predicted sample outputs
- `code/.cache/` stores reusable live-response caches for claim normalization, image review, and claim aggregation

## Approach

ClaimGuard has three layers:

1. Transcript parsing
   It extracts the most likely issue type, object part, and risk context from the customer conversation.

2. Evidence review
   It analyzes image quality locally, uses sample-image retrieval as a deterministic fallback, and optionally calls multimodal providers for richer image judgments.

3. Arbitration
   It reconciles transcript cues, per-image evidence, history flags, and evidence rules into the exact HackerRank output schema.

The ensemble path is intentionally conservative. It keeps retrieval as the floor, but upgrades to multimodal decisions when the live answer is more specific, cleaner, or better supported.

## Benchmark Snapshot

Current sample benchmark in `code/evaluation/evaluation_report.md`:

| Strategy | Exact Row Accuracy |
|---|---:|
| `text_baseline` | 25.00% |
| `retrieval` | 50.00% |
| `hybrid` | 20.00% |
| `ensemble` | 60.00% |

The current checked-in report is based on the fresh `PROMPT_VERSION=live-rerun-20260620b` benchmark path, where the ensemble keeps retrieval as a floor and only promotes live outputs when they are cleaner or better grounded than the deterministic fallback.

## Run Locally

Install dependencies:

```powershell
python -m pip install -r code/requirements.txt
```

Generate final predictions:

```powershell
python code/main.py --strategy ensemble
```

Run the evaluation workflow:

```powershell
python code/evaluation/main.py
```

Important environment variables:

- `NVIDIA_API_KEY` for the NVIDIA live path
- `OPENROUTER_API_KEY` for optional OpenRouter fallback
- `PROMPT_VERSION`
  Default is `v8` so the cache-backed benchmark remains reproducible locally. Increase it when you want to force a fresh live rerun such as `live-rerun-20260620b`.

## Output Artifacts

- Final predictions: `output.csv`
- Sample benchmark report: `code/evaluation/evaluation_report.md`
- Sample prediction files:
  - `code/evaluation/retrieval_sample_predictions.csv`
  - `code/evaluation/hybrid_sample_predictions.csv`
  - `code/evaluation/ensemble_sample_predictions.csv`
- Static report explorer: `code/evaluation/report/index.html`

## Submission Notes

- `output.csv` is generated with the exact required column order
- the evaluation workflow compares more than two strategies as required
- the repo includes a project-specific README, runnable code, cached live artifacts, and benchmark outputs
- no secrets are committed in the repository
- the latest fresh benchmark report is `code/evaluation/evaluation_report.md`

Final submission checklist:

- `submission/code.zip`
- `submission/output.csv`
- `%USERPROFILE%\\hackerrank_orchestrate\\log.txt` as the chat transcript artifact referenced by `AGENTS.md`

Chat transcript logging still follows the organizer instructions in `AGENTS.md`. Submit the generated `log.txt` from the configured path alongside `code.zip` and `output.csv`.
