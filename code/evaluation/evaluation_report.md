# Evaluation Report

## Strategy Comparison

| Strategy | Exact Row Accuracy | Notes |
|---|---:|---|
| text_baseline | 20.00% | Transcript parsing with image quality checks only. |
| retrieval | 30.00% | Offline retrieval plus rule arbitration. |
| hybrid | 20.00% | Live per-image multimodal review plus claim aggregation when keys are present, otherwise retrieval plus rule arbitration. |

Selected final strategy: `retrieval`

## Best Strategy Field Accuracy

| Field | Accuracy |
|---|---:|
| evidence_standard_met | 85.00% |
| risk_flags | 40.00% |
| issue_type | 75.00% |
| object_part | 95.00% |
| claim_status | 95.00% |
| supporting_image_ids | 65.00% |
| valid_image | 90.00% |
| severity | 90.00% |

## Operational Analysis

- Approximate model calls for full processing: `20`
- Approximate input token usage: `18000`
- Approximate output token usage: `5000`
- Number of images processed during this run: `87`
- Cost assumption: NVIDIA Build developer endpoint assumed to be free during hackathon development; OpenRouter fallback cost not incurred unless explicitly enabled.
- Latency/runtime note: Live mode performs one claim-normalization call, one image-review call per image, and one text-only aggregation call per row. Oversized images are compressed for inline transport and retried with provider/model fallbacks when needed.
- TPM/RPM note: The pipeline is sequential by default, cache-aware, and retries across current NIM-compatible multimodal models before dropping to the offline retrieval fallback.
