# Evaluation Report

## Strategy Comparison

| Strategy | Exact Row Accuracy | Notes |
|---|---:|---|
| text_baseline | 25.00% | Transcript parsing with image quality checks only. |
| retrieval | 50.00% | Offline retrieval plus rule arbitration. |
| hybrid | 20.00% | Live per-image multimodal review plus claim aggregation when keys are present, otherwise retrieval plus rule arbitration. |
| ensemble | 60.00% | Retrieval-first arbitration that promotes live multimodal outputs only when the live decision is cleaner or more grounded than the fallback. |

Selected final strategy: `ensemble`

## Best Strategy Field Accuracy

| Field | Accuracy |
|---|---:|
| evidence_standard_met | 90.00% |
| risk_flags | 70.00% |
| issue_type | 80.00% |
| object_part | 95.00% |
| claim_status | 95.00% |
| supporting_image_ids | 85.00% |
| valid_image | 90.00% |
| severity | 90.00% |

## Operational Analysis

- Approximate logical model calls for the selected strategy: `69`
- Approximate input token usage for the selected strategy: `14311`
- Approximate output token usage for the selected strategy: `12020`
- Actual uncached provider request attempts during the evaluation run: `15`
- Cache hits during the evaluation run: `123`
- Number of images processed by the selected strategy during this run: `29`
- Cost assumption: Live provider keys were available and this evaluation run issued uncached provider requests. Cache reuse may also have reduced repeated calls for already-warmed steps.
- Latency/runtime note: Ensemble mode runs the offline retrieval fallback first, then a live claim-normalization call, one live image-review call per image, and one live text-only aggregation call per row before promoting the live answer only when it is cleaner or better grounded. This run also reused cached live responses for already-computed steps.
- TPM/RPM note: The pipeline is sequential by default, cache-aware, and retries across current NIM-compatible multimodal models before dropping to the offline retrieval fallback.
