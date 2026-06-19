"""Dataclasses shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedClaim:
    claim_summary: str
    issue_type: str
    object_part: str
    claimed_parts: list[str] = field(default_factory=list)
    mentions_multiple_parts: bool = False
    prompt_injection_detected: bool = False
    confidence: float = 0.0


@dataclass
class ImageQC:
    image_path: Path
    image_id: str
    width: int
    height: int
    brightness: float
    edge_variance: float
    bright_fraction: float
    usable: bool
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class ImageReview:
    image_id: str
    claim_status: str
    visible_issue_type: str
    visible_object_part: str
    evidence_sufficient: bool
    claimed_part_visible: bool
    severity: str
    risk_flags: list[str]
    justification: str
    confidence: float


@dataclass
class Prediction:
    row: dict[str, str]
    values: dict[str, str]


@dataclass
class StrategyMetrics:
    name: str
    exact_match_accuracy: float
    field_accuracies: dict[str, float]
    row_count: int
    notes: str = ""


@dataclass
class RuntimeStats:
    provider_calls: int = 0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    images_processed: int = 0
    cache_hits: int = 0
    logical_normalize_calls: int = 0
    logical_image_review_calls: int = 0
    logical_claim_aggregate_calls: int = 0
    logical_input_tokens_estimate: int = 0
    logical_output_tokens_estimate: int = 0

    @property
    def logical_provider_calls(self) -> int:
        return self.logical_normalize_calls + self.logical_image_review_calls + self.logical_claim_aggregate_calls

    def merge(self, other: "RuntimeStats") -> None:
        self.provider_calls += other.provider_calls
        self.input_tokens_estimate += other.input_tokens_estimate
        self.output_tokens_estimate += other.output_tokens_estimate
        self.images_processed += other.images_processed
        self.cache_hits += other.cache_hits
        self.logical_normalize_calls += other.logical_normalize_calls
        self.logical_image_review_calls += other.logical_image_review_calls
        self.logical_claim_aggregate_calls += other.logical_claim_aggregate_calls
        self.logical_input_tokens_estimate += other.logical_input_tokens_estimate
        self.logical_output_tokens_estimate += other.logical_output_tokens_estimate

