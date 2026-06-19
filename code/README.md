# Multi-Modal Evidence Review Solution

This `code/` directory contains a Python-first claim-review pipeline for the HackerRank Orchestrate June 2026 challenge.

## What it does

- Parses the claim conversation into a structured damage summary.
- Reviews each submitted image with a live NVIDIA NIM / OpenRouter multimodal path when API keys are available.
- Falls back to an offline retrieval-and-rules strategy when live model access is unavailable.
- Writes exact-schema predictions to `output.csv`.
- Evaluates the sample set and generates:
  - `code/evaluation/evaluation_report.md`
  - `code/evaluation/report/index.html`

## Setup

Windows PowerShell:

```powershell
python -m pip install -r code/requirements.txt
```

Optional environment variables for the live multimodal path:

- `NVIDIA_API_KEY`
- `NVIDIA_BASE_URL` default: `https://integrate.api.nvidia.com/v1`
- `PRIMARY_MODEL` default: `stepfun-ai/step-3.7-flash`
- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` default: `https://openrouter.ai/api/v1`
- `FALLBACK_MODEL` default: `qwen/qwen2.5-vl-72b-instruct`
- `OPENROUTER_REFERER`
- `OPENROUTER_TITLE`

If no provider keys are present, the pipeline still runs using the offline retrieval fallback.

## Run predictions

```powershell
python code/main.py
```

This writes `output.csv` at the repo root.

## Run evaluation

```powershell
python code/evaluation/main.py
```

This compares a text-only baseline against the current final strategy and writes the markdown report plus the static HTML explorer.

## Notes

- `code/.cache/` stores file-backed caches for live normalization and per-image review responses.
- The pipeline is deterministic where practical: temperature is fixed to `0.0`, enum values are sanitized, and cached live responses are reused.
- The offline fallback is included to keep the project runnable even when no API keys are set locally.
