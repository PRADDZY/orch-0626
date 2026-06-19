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
- `PROMPT_VERSION` default: `v8`
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

By default the CLI uses the `ensemble` strategy. When live provider keys are available it runs retrieval plus the live multimodal path, then promotes the live answer only when that answer is clearly cleaner or better grounded than retrieval. If no live keys are present, it automatically falls back to `retrieval`.

Useful overrides:

- `python code/main.py --strategy retrieval`
- `python code/main.py --strategy hybrid`
- `python code/main.py --strategy text_baseline`

## Run evaluation

```powershell
python code/evaluation/main.py
```

This compares the text baseline, retrieval fallback, live hybrid path, and the gated ensemble strategy, then writes the markdown report plus the static HTML explorer.

## Notes

- `code/.cache/` stores file-backed caches for live normalization, per-image review responses, and claim-level aggregation results.
- The default `PROMPT_VERSION=v8` is intentional for local reproducibility because the repo already contains a compatible cache-backed live benchmark profile. Bump the version when you want to force a fully fresh live rerun.
- The pipeline is deterministic where practical: temperature is fixed to `0.0`, enum values are sanitized, and cached live responses are reused.
- The live NIM path uses current working multimodal models, retries across model candidates, and compresses oversized images for inline transport when raw uploads are too large or asset mounting is flaky.
- In restricted environments where outbound provider access is blocked, the hybrid and ensemble paths still reuse compatible cached live responses when present and otherwise fall back cleanly to retrieval.
- Evaluation compares `retrieval`, `hybrid`, and `ensemble` on the updated sample set and selects the strongest strategy for the generated report artifacts.
- The offline fallback is included to keep the project runnable even when no API keys are set locally.
