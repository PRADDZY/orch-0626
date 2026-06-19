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
- `PRIMARY_MODEL` default: `google/gemma-4-31b-it`
- `NIM_FALLBACK_MODELS` default: `meta/llama-4-maverick-17b-128e-instruct,nvidia/nemotron-nano-12b-v2-vl`
- `NIM_INLINE_IMAGE_LIMIT_KB` default: `180`
- `NIM_POLL_TIMEOUT_SECONDS` default: `45`
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

By default the CLI uses the retrieval strategy for submission safety. Use `python code/main.py --strategy hybrid` to force the live NIM-first path for demos or experimentation.

## Run evaluation

```powershell
python code/evaluation/main.py
```

This compares a text-only baseline against the current final strategy and writes the markdown report plus the static HTML explorer.

## Notes

- `code/.cache/` stores file-backed caches for live normalization, per-image review responses, and claim-level aggregation results.
- The pipeline is deterministic where practical: temperature is fixed to `0.0`, enum values are sanitized, and cached live responses are reused.
- The live NIM path uses current working multimodal models, retries across model candidates, and compresses oversized images for inline transport when raw uploads are too large or asset mounting is flaky.
- Evaluation compares `retrieval` and `hybrid` on the updated sample set and selects the stronger strategy for the generated report artifacts.
- The offline fallback is included to keep the project runnable even when no API keys are set locally.
