# Evaluation Report

## Strategy Comparison

| Strategy | Exact Row Accuracy | Notes |
|---|---:|---|
| text_baseline | 10.00% | Transcript parsing with image quality checks only. |
| retrieval | 20.00% | Live multimodal review when keys are present, otherwise retrieval plus rule arbitration. |

Selected final strategy: `retrieval`

## Best Strategy Field Accuracy

| Field | Accuracy |
|---|---:|
| evidence_standard_met | 90.00% |
| risk_flags | 35.00% |
| issue_type | 80.00% |
| object_part | 95.00% |
| claim_status | 95.00% |
| supporting_image_ids | 65.00% |
| valid_image | 90.00% |
| severity | 90.00% |

## Operational Analysis

- Approximate model calls for full processing: `0`
- Approximate input token usage: `0`
- Approximate output token usage: `0`
- Number of images processed during this run: `58`
- Cost assumption: Live provider keys were not available in the local environment, so the offline retrieval fallback was executed at zero API cost.
- Latency/runtime note: Live mode is designed for one normalization call plus one image-review call per image, with file-backed caching to avoid repeats.
- TPM/RPM note: The pipeline is sequential by default, cache-aware, and can be batched later if provider RPM limits become visible during a live run.
