# Evaluation Report

## Strategy Comparison

| Strategy | Exact Row Accuracy | Notes |
|---|---:|---|
| text_baseline | 20.00% | Transcript parsing with image quality checks only. |
| retrieval | 30.00% | Offline retrieval plus rule arbitration. |
| hybrid | 30.00% | Live per-image multimodal review plus claim aggregation when keys are present, otherwise retrieval plus rule arbitration. |
| ensemble | 35.00% | Retrieval-first arbitration that promotes live multimodal outputs only when the live decision is cleaner or more grounded than the fallback. |

Selected final strategy: `ensemble`

## Best Strategy Field Accuracy

| Field | Accuracy |
|---|---:|
| evidence_standard_met | 85.00% |
| risk_flags | 45.00% |
| issue_type | 75.00% |
| object_part | 95.00% |
| claim_status | 95.00% |
| supporting_image_ids | 70.00% |
| valid_image | 90.00% |
| severity | 90.00% |

## Operational Analysis

- Approximate model calls for full processing: `64`
- Approximate input token usage: `57600`
- Approximate output token usage: `16000`
- Number of images processed during this run: `29`
- Cost assumption: NVIDIA Build developer endpoint assumed to be free during hackathon development; OpenRouter fallback cost not incurred unless explicitly enabled.
- Latency/runtime note: Ensemble mode runs the offline retrieval fallback first, then a live claim-normalization call, one live image-review call per image, and one live text-only aggregation call per row before promoting the live answer only when it is cleaner or better grounded.
- TPM/RPM note: The pipeline is sequential by default, cache-aware, and retries across current NIM-compatible multimodal models before dropping to the offline retrieval fallback.
