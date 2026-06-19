"""Core pipeline, fallbacks, and CSV helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat

from .config import AppConfig
from .constants import (
    CLAIM_STATUSES,
    DEFAULT_SEVERITY_BY_ISSUE,
    ISSUE_KEYWORDS,
    ISSUE_TYPES,
    OBJECT_PARTS,
    OUTPUT_COLUMNS,
    PART_KEYWORDS,
    PROMPT_INJECTION_PATTERNS,
    RISK_FLAG_ORDER,
    RISK_FLAGS,
    SEVERITIES,
)
from .models import ImageQC, ImageReview, ParsedClaim, Prediction, RuntimeStats, StrategyMetrics
from .providers import OpenAICompatibleProvider


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def lower_alnum(text: str) -> str:
    return normalize_spaces(re.sub(r"[^a-z0-9\s]", " ", text.lower()))


def resolve_dataset_path(repo_root: Path, relative_path: str) -> Path:
    direct = repo_root / relative_path
    if direct.exists():
        return direct
    return repo_root / "dataset" / relative_path


def image_id_from_path(relative_path: str) -> str:
    return Path(relative_path).stem


def to_bool_text(value: bool) -> str:
    return "true" if value else "false"


def choose_severity(issue_type: str) -> str:
    return DEFAULT_SEVERITY_BY_ISSUE.get(issue_type, "unknown")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def coerce_list(value: Any) -> list[str]:
    if isinstance(value, str):
        if not value.strip() or value.strip().lower() == "none":
            return []
        return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_issue_value(value: str, fallback: str = "unknown") -> str:
    normalized = lower_alnum(value)
    direct = normalized.replace(" ", "_")
    if direct in ISSUE_TYPES:
        return direct
    for label, phrases in ISSUE_KEYWORDS.items():
        if label in normalized or any(phrase in normalized for phrase in phrases):
            return label
    if any(term in normalized for term in {"no damage", "no visible damage", "intact", "undamaged", "no issue"}):
        return "none"
    return fallback if fallback in ISSUE_TYPES else "unknown"


def normalize_object_part_value(value: str, claim_object: str, fallback: str = "unknown") -> str:
    normalized = lower_alnum(value)
    direct = normalized.replace(" ", "_")
    allowed_parts = OBJECT_PARTS[claim_object]
    if direct in allowed_parts:
        return direct
    options = {
        label: [label.replace("_", " "), *phrases]
        for label, phrases in PART_KEYWORDS.get(claim_object, {}).items()
    }
    hits = keyword_match(normalized, options)
    if hits:
        return hits[0]
    return fallback if fallback in allowed_parts else "unknown"


def normalize_claim_status_value(value: str, fallback: str = "not_enough_information") -> str:
    normalized = lower_alnum(value)
    direct = normalized.replace(" ", "_")
    if direct in CLAIM_STATUSES:
        return direct
    if any(term in normalized for term in {"support", "confirmed", "match"}):
        return "supported"
    if any(term in normalized for term in {"contradict", "false", "wrong", "invalid", "mismatch"}):
        return "contradicted"
    return fallback


def normalize_severity_value(value: str, issue_type: str, claim_status: str, risk_flags: list[str]) -> str:
    normalized = lower_alnum(value).replace(" ", "_")
    if normalized in SEVERITIES:
        severity = normalized
    else:
        severity = choose_severity(issue_type)
    if claim_status == "not_enough_information":
        return "unknown"
    if claim_status == "contradicted" and issue_type == "none":
        return "none"
    if claim_status == "contradicted" and "wrong_object" in risk_flags and severity == "unknown":
        return "low"
    return severity


def extract_customer_text(transcript: str) -> str:
    chunks = []
    for segment in transcript.split("|"):
        segment = segment.strip()
        if ":" in segment:
            speaker, content = segment.split(":", 1)
            if speaker.strip().lower() in {"customer", "claimant", "user"}:
                chunks.append(content.strip())
    if chunks:
        return normalize_spaces(" ".join(chunks))
    return normalize_spaces(transcript)


def keyword_match(text: str, options: dict[str, list[str]]) -> list[str]:
    matches: list[tuple[int, str]] = []
    for label, phrases in options.items():
        hits = 0
        for phrase in phrases:
            if phrase in text:
                hits += max(1, len(phrase.split()))
        if hits:
            matches.append((hits, label))
    matches.sort(reverse=True)
    return [label for _, label in matches]


def parse_claim_heuristic(user_claim: str, claim_object: str) -> ParsedClaim:
    customer_text = extract_customer_text(user_claim)
    text = lower_alnum(customer_text)
    issue_hits = keyword_match(text, ISSUE_KEYWORDS)
    part_hits = keyword_match(text, PART_KEYWORDS.get(claim_object, {}))
    injection = any(phrase in text for phrase in PROMPT_INJECTION_PATTERNS)
    multiple_parts = (
        len(part_hits) > 1
        and any(token in f" {text} " for token in (" and ", " both ", " multiple ", " two ", " plus "))
    ) or "two things" in text or "multiple parts" in text
    issue_type = issue_hits[0] if issue_hits else "unknown"
    object_part = part_hits[0] if part_hits else "unknown"
    summary_bits = [claim_object]
    if issue_type != "unknown":
        summary_bits.append(issue_type)
    if object_part != "unknown":
        summary_bits.append(object_part)
    if not summary_bits:
        summary_bits.append("unclear damage claim")
    return ParsedClaim(
        claim_summary=normalize_spaces(" ".join(summary_bits)),
        issue_type=issue_type,
        object_part=object_part,
        claimed_parts=part_hits[:3],
        mentions_multiple_parts=multiple_parts,
        prompt_injection_detected=injection,
        confidence=0.55 if issue_hits or part_hits else 0.2,
    )


def average_hash(image: Image.Image, size: int = 8) -> str:
    reduced = ImageOps.grayscale(image).resize((size, size))
    pixels = list(reduced.getdata())
    threshold = sum(pixels) / len(pixels)
    return "".join("1" if value >= threshold else "0" for value in pixels)


def difference_hash(image: Image.Image, size: int = 8) -> str:
    reduced = ImageOps.grayscale(image).resize((size + 1, size))
    pixels = list(reduced.getdata())
    bits: list[str] = []
    for row in range(size):
        row_offset = row * (size + 1)
        for col in range(size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            bits.append("1" if left > right else "0")
    return "".join(bits)


def hamming_similarity(a: str, b: str) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    same = sum(1 for left, right in zip(a, b) if left == right)
    return same / len(a)


def color_signature(image: Image.Image) -> tuple[float, float, float]:
    thumb = image.resize((4, 4)).convert("RGB")
    stat = ImageStat.Stat(thumb)
    return tuple(float(value) for value in stat.mean)


def color_similarity(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    distance = math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))
    return max(0.0, 1.0 - distance / 441.6729559300637)


def analyze_image_qc(image_path: Path, image_id: str) -> ImageQC:
    try:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            gray = ImageOps.grayscale(image)
            width, height = image.size
            brightness = float(ImageStat.Stat(gray).mean[0])
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_variance = float(ImageStat.Stat(edges).var[0])
            pixels = list(gray.getdata())
            bright_fraction = sum(1 for pixel in pixels if pixel >= 245) / max(1, len(pixels))
            risk_flags: list[str] = []
            if edge_variance < 140:
                risk_flags.append("blurry_image")
            if brightness < 28 or bright_fraction > 0.35:
                risk_flags.append("low_light_or_glare")
            if min(width, height) < 120:
                risk_flags.append("cropped_or_obstructed")
            return ImageQC(
                image_path=image_path,
                image_id=image_id,
                width=width,
                height=height,
                brightness=brightness,
                edge_variance=edge_variance,
                bright_fraction=bright_fraction,
                usable=True,
                risk_flags=risk_flags,
            )
    except Exception:
        return ImageQC(
            image_path=image_path,
            image_id=image_id,
            width=0,
            height=0,
            brightness=0.0,
            edge_variance=0.0,
            bright_fraction=0.0,
            usable=False,
            risk_flags=["cropped_or_obstructed"],
        )


class SampleMatcher:
    """Cheap offline retrieval fallback using sample exemplars."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.entries = self._build_entries()

    def _image_features(self, image_path: Path) -> dict[str, Any]:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            return {
                "ahash": average_hash(image),
                "dhash": difference_hash(image),
                "color": color_signature(image),
            }

    def _build_entries(self) -> list[dict[str, Any]]:
        sample_path = self.repo_root / "dataset" / "sample_claims.csv"
        rows = read_csv_rows(sample_path)
        entries: list[dict[str, Any]] = []
        for row in rows:
            parsed = parse_claim_heuristic(row["user_claim"], row["claim_object"])
            relative_paths = [item.strip() for item in row["image_paths"].split(";") if item.strip()]
            features = []
            for relative_path in relative_paths:
                image_path = resolve_dataset_path(self.repo_root, relative_path)
                try:
                    features.append(self._image_features(image_path))
                except Exception:
                    continue
            entries.append(
                {
                    "claim_object": row["claim_object"],
                    "parsed": parsed,
                    "row": row,
                    "features": features,
                }
            )
        return entries

    def score(
        self,
        target_features: list[tuple[int, dict[str, Any]]],
        exemplar: dict[str, Any],
        parsed: ParsedClaim,
    ) -> tuple[float, int | None]:
        if not target_features or not exemplar["features"]:
            image_score = 0.0
            best_target_index: int | None = None
        else:
            image_score = 0.0
            best_target_index = None
            for target_index, target in target_features:
                for candidate in exemplar["features"]:
                    score = (
                        hamming_similarity(target["ahash"], candidate["ahash"]) * 0.45
                        + hamming_similarity(target["dhash"], candidate["dhash"]) * 0.35
                        + color_similarity(target["color"], candidate["color"]) * 0.20
                    )
                    if score > image_score:
                        image_score = score
                        best_target_index = target_index
        issue_bonus = 0.18 if parsed.issue_type != "unknown" and parsed.issue_type == exemplar["row"]["issue_type"] else 0.0
        part_bonus = 0.14 if parsed.object_part != "unknown" and parsed.object_part == exemplar["row"]["object_part"] else 0.0
        return image_score * 0.68 + issue_bonus + part_bonus, best_target_index

    def best_match(
        self,
        claim_object: str,
        parsed: ParsedClaim,
        image_paths: list[Path],
    ) -> dict[str, Any] | None:
        target_features = []
        for index, image_path in enumerate(image_paths):
            try:
                target_features.append((index, self._image_features(image_path)))
            except Exception:
                continue
        best: dict[str, Any] | None = None
        best_score = -1.0
        for entry in self.entries:
            if entry["claim_object"] != claim_object:
                continue
            score, matched_target_index = self.score(target_features, entry, parsed)
            if score > best_score:
                best = {"entry": entry, "score": score, "matched_target_index": matched_target_index}
                best_score = score
        return best


class ClaimReviewer:
    """End-to-end claim reviewer with a live-model path and an offline fallback."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.history_by_user = {
            row["user_id"]: row
            for row in read_csv_rows(self.config.repo_root / "dataset" / "user_history.csv")
        }
        self.requirements_rows = read_csv_rows(self.config.repo_root / "dataset" / "evidence_requirements.csv")
        self.sample_matcher = SampleMatcher(self.config.repo_root)
        self.runtime_stats = RuntimeStats()
        self.providers = [
            OpenAICompatibleProvider(provider_config)
            for provider_config in (self.config.primary_provider, self.config.fallback_provider)
            if provider_config.enabled
        ]

    def _provider_chain(self, method_name: str, **kwargs: Any) -> dict[str, Any] | None:
        for provider in self.providers:
            try:
                payload = getattr(provider, method_name)(**kwargs)
            except Exception:
                self.runtime_stats.provider_calls += 1
                continue
            self.runtime_stats.provider_calls += 1
            if payload:
                return payload
        return None

    def _cache_path(self, namespace: str, cache_key: str) -> Path:
        directory = self.config.cache_dir / namespace
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{cache_key}.json"

    def _load_cache(self, namespace: str, cache_key: str) -> dict[str, Any] | None:
        cache_path = self._cache_path(namespace, cache_key)
        if not cache_path.exists():
            return None
        self.runtime_stats.cache_hits += 1
        return json.loads(cache_path.read_text(encoding="utf-8"))

    def _save_cache(self, namespace: str, cache_key: str, payload: dict[str, Any]) -> None:
        cache_path = self._cache_path(namespace, cache_key)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _history_risk_flags(self, user_id: str) -> list[str]:
        row = self.history_by_user.get(user_id)
        if not row:
            return []
        flags = [flag for flag in row["history_flags"].split(";") if flag and flag != "none"]
        return [flag for flag in flags if flag in RISK_FLAGS]

    def _evidence_rules(self, claim_object: str, issue_type: str) -> list[str]:
        rows = []
        for item in self.requirements_rows:
            if item["claim_object"] not in {"all", claim_object}:
                continue
            applies = item["applies_to"].lower()
            if item["claim_object"] == "all" or issue_type == "unknown" or issue_type.replace("_", " ") in applies:
                rows.append(item["minimum_image_evidence"])
                continue
            if claim_object == "car" and issue_type in {"dent", "scratch"} and "dent or scratch" in applies:
                rows.append(item["minimum_image_evidence"])
            elif claim_object == "package" and issue_type in {"water_damage", "stain"} and "water" in applies:
                rows.append(item["minimum_image_evidence"])
            elif claim_object == "package" and issue_type in {"torn_packaging", "crushed_packaging"} and "seal damage" in applies:
                rows.append(item["minimum_image_evidence"])
        general_rules = [item["minimum_image_evidence"] for item in self.requirements_rows if item["claim_object"] == "all"]
        return list(dict.fromkeys(general_rules + rows))

    def _split_semicolon_value(self, value: str) -> list[str]:
        return [item.strip() for item in value.split(";") if item.strip() and item.strip() != "none"]

    def _prediction_flags(self, prediction: Prediction) -> set[str]:
        return set(self._split_semicolon_value(prediction.values["risk_flags"]))

    def _prediction_supporting_ids(self, prediction: Prediction) -> list[str]:
        return self._split_semicolon_value(prediction.values["supporting_image_ids"])

    def _is_clean_live_decision(self, prediction: Prediction) -> bool:
        flags = self._prediction_flags(prediction)
        if prediction.values["claim_status"] not in {"supported", "contradicted"}:
            return False
        if prediction.values["issue_type"] == "missing_part" or prediction.values["object_part"] == "contents":
            return False
        if prediction.values["evidence_standard_met"] != "true":
            return False
        if prediction.values["valid_image"] != "true":
            return False
        blocking_flags = {
            "wrong_object",
            "claim_mismatch",
            "damage_not_visible",
            "possible_manipulation",
            "non_original_image",
            "text_instruction_present",
            "manual_review_required",
        }
        return not flags.intersection(blocking_flags)

    def _choose_ensemble_prediction(self, retrieval_prediction: Prediction, hybrid_prediction: Prediction) -> Prediction:
        retrieval_values = retrieval_prediction.values
        hybrid_values = hybrid_prediction.values
        retrieval_flags = self._prediction_flags(retrieval_prediction)
        hybrid_flags = self._prediction_flags(hybrid_prediction)

        same_core_decision = all(
            retrieval_values[field] == hybrid_values[field]
            for field in ("claim_status", "issue_type", "object_part", "severity", "valid_image", "evidence_standard_met")
        )
        if same_core_decision:
            if hybrid_flags < retrieval_flags:
                return hybrid_prediction
            retrieval_support = self._prediction_supporting_ids(retrieval_prediction)
            hybrid_support = self._prediction_supporting_ids(hybrid_prediction)
            if hybrid_flags == retrieval_flags and hybrid_support and len(hybrid_support) == 1 and len(retrieval_support) > 1:
                return hybrid_prediction

        if retrieval_values["claim_status"] == "not_enough_information" and self._is_clean_live_decision(hybrid_prediction):
            return hybrid_prediction

        return retrieval_prediction

    def _refine_ensemble_prediction(self, prediction: Prediction, live_reviews: list[ImageReview]) -> Prediction:
        flags = self._prediction_flags(prediction)
        refined_values = dict(prediction.values)

        if prediction.values["claim_status"] == "supported":
            matching_supported_reviews = [
                review
                for review in live_reviews
                if review.claim_status == "supported"
                and review.evidence_sufficient
                and review.visible_issue_type == prediction.values["issue_type"]
                and review.visible_object_part == prediction.values["object_part"]
            ]
            if len(matching_supported_reviews) == 1:
                refined_values["supporting_image_ids"] = matching_supported_reviews[0].image_id

        if (
            prediction.values["claim_status"] == "contradicted"
            and prediction.values["issue_type"] == "none"
            and any(
                review.claim_status == "contradicted"
                and review.evidence_sufficient
                and review.claimed_part_visible
                and review.visible_object_part == prediction.values["object_part"]
                for review in live_reviews
            )
        ):
            flags.add("damage_not_visible")
            flags.discard("claim_mismatch")

        if prediction.values["claim_status"] == "not_enough_information":
            insufficiency_flags = {
                flag
                for review in live_reviews
                for flag in review.risk_flags
                if flag in {"wrong_angle", "damage_not_visible", "cropped_or_obstructed"}
            }
            flags.update(insufficiency_flags)
            if not any(review.claimed_part_visible for review in live_reviews):
                refined_values["evidence_standard_met"] = "false"
                refined_values["issue_type"] = "unknown"
                refined_values["supporting_image_ids"] = "none"

        if any("text_instruction_present" in review.risk_flags for review in live_reviews):
            flags.add("text_instruction_present")
            flags.add("manual_review_required")
            if prediction.values["claim_status"] == "contradicted" and prediction.values["issue_type"] == "none":
                flags.discard("claim_mismatch")
                flags.add("damage_not_visible")
        refined_values["risk_flags"] = ";".join(flag for flag in RISK_FLAG_ORDER if flag in flags) if flags else "none"
        return Prediction(row=prediction.row, values=refined_values)

    def _aggregate_live_reviews_fallback(
        self,
        row: dict[str, str],
        parsed: ParsedClaim,
        qcs: list[ImageQC],
        reviews: list[ImageReview],
    ) -> Prediction | None:
        has_live_signal = any(
            review.evidence_sufficient
            or review.claim_status != "not_enough_information"
            or review.visible_issue_type != "unknown"
            or review.confidence > 0.35
            or review.risk_flags
            for review in reviews
        )
        if not has_live_signal:
            return None

        history_flags = self._history_risk_flags(row["user_id"])
        supported = [
            review
            for review in reviews
            if review.claim_status == "supported" and (review.evidence_sufficient or review.visible_issue_type != "unknown")
        ]
        contradicted = [
            review
            for review in reviews
            if review.claim_status == "contradicted" and (review.evidence_sufficient or review.visible_issue_type != "unknown")
        ]
        insufficient = [review for review in reviews if review.claim_status == "not_enough_information"]

        if supported:
            chosen = max(supported, key=lambda review: (review.evidence_sufficient, review.claimed_part_visible, review.confidence))
        elif contradicted:
            chosen = max(contradicted, key=lambda review: (review.evidence_sufficient, review.confidence))
        elif insufficient:
            chosen = max(insufficient, key=lambda review: review.confidence)
        else:
            return None

        claim_status = chosen.claim_status
        issue_type = chosen.visible_issue_type if chosen.visible_issue_type != "unknown" else parsed.issue_type
        object_part = chosen.visible_object_part if chosen.visible_object_part != "unknown" else parsed.object_part
        risk_flags: list[str] = []
        for qc in qcs:
            for flag in qc.risk_flags:
                if flag not in risk_flags:
                    risk_flags.append(flag)
        for review in reviews:
            for flag in review.risk_flags:
                if flag not in risk_flags:
                    risk_flags.append(flag)
        for flag in history_flags:
            if flag not in risk_flags:
                risk_flags.append(flag)
        if parsed.prompt_injection_detected and "text_instruction_present" not in risk_flags:
            risk_flags.append("text_instruction_present")

        corroborated_non_original = (
            sum(1 for review in reviews if "non_original_image" in review.risk_flags) >= 2
            or any(
                "non_original_image" in review.risk_flags
                and any(flag in review.risk_flags for flag in {"possible_manipulation", "text_instruction_present"})
                for review in reviews
            )
        )
        if claim_status == "supported" and not contradicted and not corroborated_non_original:
            risk_flags = [flag for flag in risk_flags if flag not in {"non_original_image", "claim_mismatch"}]
        if claim_status != "not_enough_information":
            risk_flags = [flag for flag in risk_flags if flag != "damage_not_visible"]
        elif not any(review.claimed_part_visible for review in reviews) and "damage_not_visible" not in risk_flags:
            risk_flags.append("damage_not_visible")

        if claim_status == "supported" and any(
            flag in risk_flags for flag in {"wrong_object", "claim_mismatch", "possible_manipulation", "text_instruction_present"}
        ):
            claim_status = "not_enough_information"
        if claim_status == "supported" and "non_original_image" in risk_flags and corroborated_non_original:
            claim_status = "not_enough_information"

        evidence_standard_met = (
            claim_status != "not_enough_information"
            and any(review.evidence_sufficient for review in reviews if review.claim_status == chosen.claim_status)
            and any(qc.usable for qc in qcs)
        )
        valid_image = any(qc.usable for qc in qcs)
        if any(flag in risk_flags for flag in {"possible_manipulation", "text_instruction_present"}):
            valid_image = False
        elif "non_original_image" in risk_flags and corroborated_non_original and claim_status != "supported":
            valid_image = False

        manual_review_required = False
        if any(flag in risk_flags for flag in {"wrong_object", "possible_manipulation", "text_instruction_present"}):
            manual_review_required = True
        if "claim_mismatch" in risk_flags and claim_status != "supported":
            manual_review_required = True
        if "non_original_image" in risk_flags and corroborated_non_original:
            manual_review_required = True
        if "user_history_risk" in risk_flags:
            manual_review_required = True
        if supported and contradicted:
            manual_review_required = True
        if manual_review_required and "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        if not manual_review_required:
            risk_flags = [flag for flag in risk_flags if flag != "manual_review_required"]

        issue_type = issue_type if issue_type in ISSUE_TYPES else "unknown"
        object_part = object_part if object_part in OBJECT_PARTS[row["claim_object"]] else "unknown"
        severity = choose_severity(issue_type)
        if claim_status == "contradicted" and issue_type == "none":
            severity = "none"
        if claim_status == "not_enough_information":
            severity = "unknown"

        supporting_image_ids = "none"
        if claim_status != "not_enough_information":
            supporting_image_ids = chosen.image_id if chosen.image_id != "none" else "none"

        if evidence_standard_met:
            evidence_reason = (
                f"The claimed {object_part.replace('_', ' ')} is visible clearly enough in the submitted image set to evaluate the claim."
            )
        else:
            evidence_reason = (
                f"The submitted images do not show the claimed {parsed.object_part.replace('_', ' ')} clearly enough for a reliable decision."
                if parsed.object_part != "unknown"
                else "The submitted images do not provide enough clear evidence to evaluate the claim."
            )

        if claim_status == "supported":
            status_justification = f"{chosen.justification} Supporting images: {supporting_image_ids}."
        elif claim_status == "contradicted":
            status_justification = f"{chosen.justification} The available images support a contradiction rather than the stated claim."
        else:
            status_justification = f"{chosen.justification} The image set is not strong enough for a definitive confirmation or contradiction."

        values = {
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            "evidence_standard_met": to_bool_text(evidence_standard_met),
            "evidence_standard_met_reason": normalize_spaces(evidence_reason),
            "risk_flags": ";".join(flag for flag in RISK_FLAG_ORDER if flag in risk_flags) if risk_flags else "none",
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": normalize_spaces(status_justification),
            "supporting_image_ids": supporting_image_ids,
            "valid_image": to_bool_text(valid_image),
            "severity": severity,
        }
        return Prediction(row=row, values=values)

    def _provider_normalize_claim(self, row: dict[str, str], parsed: ParsedClaim) -> ParsedClaim:
        if not self.providers:
            return parsed
        logical_input_tokens = max(80, len(row["user_claim"]) // 3)
        self.runtime_stats.logical_normalize_calls += 1
        self.runtime_stats.logical_input_tokens_estimate += logical_input_tokens
        self.runtime_stats.logical_output_tokens_estimate += 120
        allowed_parts = sorted(OBJECT_PARTS[row["claim_object"]])
        allowed_issues = sorted(ISSUE_TYPES)
        cache_key = stable_hash(f"{self.config.prompt_version}|normalize|{row['claim_object']}|{row['user_claim']}")
        cached = self._load_cache("normalize", cache_key)
        payload = cached
        if payload is None:
            payload = self._provider_chain(
                "normalize_claim",
                transcript=row["user_claim"],
                claim_object=row["claim_object"],
                allowed_parts=allowed_parts,
                allowed_issues=allowed_issues,
            )
            if payload:
                self._save_cache("normalize", cache_key, payload)
                self.runtime_stats.input_tokens_estimate += logical_input_tokens
                self.runtime_stats.output_tokens_estimate += 120
        if not payload:
            return parsed
        issue_type = payload.get("issue_type", parsed.issue_type)
        if issue_type not in ISSUE_TYPES:
            issue_type = parsed.issue_type
        object_part = payload.get("object_part", parsed.object_part)
        if object_part not in OBJECT_PARTS[row["claim_object"]]:
            object_part = parsed.object_part
        claimed_parts = [
            part
            for part in coerce_list(payload.get("claimed_parts", parsed.claimed_parts))
            if part in OBJECT_PARTS[row["claim_object"]]
        ]
        return ParsedClaim(
            claim_summary=normalize_spaces(payload.get("claim_summary", parsed.claim_summary)),
            issue_type=issue_type,
            object_part=object_part,
            claimed_parts=claimed_parts or parsed.claimed_parts,
            mentions_multiple_parts=bool(payload.get("mentions_multiple_parts", parsed.mentions_multiple_parts)),
            prompt_injection_detected=bool(payload.get("prompt_injection_detected", parsed.prompt_injection_detected)),
            confidence=coerce_float(payload.get("confidence"), parsed.confidence or 0.0),
        )

    def _provider_review_image(
        self,
        row: dict[str, str],
        parsed: ParsedClaim,
        qc: ImageQC,
    ) -> ImageReview:
        if not self.providers:
            raise RuntimeError("provider review requested without any providers")
        self.runtime_stats.logical_image_review_calls += 1
        self.runtime_stats.logical_input_tokens_estimate += 200
        self.runtime_stats.logical_output_tokens_estimate += 180
        cache_key = stable_hash(
            f"{self.config.prompt_version}|image|{row['claim_object']}|{parsed.claim_summary}|{qc.image_path}|{qc.edge_variance:.1f}"
        )
        cached = self._load_cache("image_review", cache_key)
        payload = cached
        if payload is None:
            payload = self._provider_chain(
                "review_image",
                image_path=qc.image_path,
                claim_object=row["claim_object"],
                claim_summary=parsed.claim_summary,
                issue_type=parsed.issue_type,
                object_part=parsed.object_part,
                evidence_rules=self._evidence_rules(row["claim_object"], parsed.issue_type),
            )
            if payload:
                self._save_cache("image_review", cache_key, payload)
                self.runtime_stats.input_tokens_estimate += 200
                self.runtime_stats.output_tokens_estimate += 180
        if not payload:
            return ImageReview(
                image_id=qc.image_id,
                claim_status="not_enough_information",
                visible_issue_type="unknown",
                visible_object_part=parsed.object_part if parsed.object_part != "unknown" else "unknown",
                evidence_sufficient=False,
                claimed_part_visible=False,
                severity="unknown",
                risk_flags=list(qc.risk_flags),
                justification="The live multimodal reviewer could not return a stable result for this image.",
                confidence=0.1,
            )
        risk_flags = [flag for flag in payload.get("risk_flags", []) if flag in RISK_FLAGS and flag != "none"]
        risk_flags.extend(flag for flag in qc.risk_flags if flag not in risk_flags)
        observed_object_raw = lower_alnum(str(payload.get("observed_object_type", row["claim_object"])))
        if "car" in observed_object_raw or "bumper" in observed_object_raw or "hood" in observed_object_raw:
            observed_object_type = "car"
        elif any(token in observed_object_raw for token in {"laptop", "keyboard", "screen", "trackpad", "hinge"}):
            observed_object_type = "laptop"
        elif any(token in observed_object_raw for token in {"package", "box", "parcel", "shipping"}):
            observed_object_type = "package"
        elif observed_object_raw in {"other", "unknown"}:
            observed_object_type = observed_object_raw
        else:
            observed_object_type = "other"
        if observed_object_type not in {row["claim_object"], "unknown"} and "wrong_object" not in risk_flags:
            risk_flags.append("wrong_object")
        claim_status = payload.get("claim_status", "not_enough_information")
        if claim_status not in CLAIM_STATUSES:
            claim_status = "not_enough_information"
        if "wrong_object" in risk_flags and claim_status == "supported":
            claim_status = "contradicted"
        issue_type = payload.get("visible_issue_type") or payload.get("visible_issue_issue_type") or payload.get("issue_type", "unknown")
        if issue_type not in ISSUE_TYPES:
            issue_type = "unknown"
        object_part = payload.get("visible_object_part", "unknown")
        if object_part not in OBJECT_PARTS[row["claim_object"]]:
            object_part = "unknown"
        if "wrong_object" in risk_flags:
            issue_type = "unknown"
            object_part = "unknown"
        severity = payload.get("severity", choose_severity(issue_type))
        if severity == "severe":
            severity = "high"
        if severity not in SEVERITIES:
            severity = choose_severity(issue_type)
        if claim_status == "contradicted" and "wrong_object" in risk_flags:
            severity = "low"
        return ImageReview(
            image_id=qc.image_id,
            claim_status=claim_status,
            visible_issue_type=issue_type,
            visible_object_part=object_part,
            evidence_sufficient=bool(payload.get("evidence_sufficient", False)),
            claimed_part_visible=bool(payload.get("claimed_part_visible", False)),
            severity=severity,
            risk_flags=risk_flags,
            justification=normalize_spaces(payload.get("justification", "Visual evidence was reviewed.")),
            confidence=coerce_float(payload.get("confidence"), 0.5),
        )

    def _provider_predict_row(
        self,
        row: dict[str, str],
        parsed: ParsedClaim,
        qcs: list[ImageQC],
        reviews: list[ImageReview],
    ) -> Prediction | None:
        if not self.providers:
            return None
        logical_input_tokens = max(250, len(row["user_claim"]) // 2 + 90 * max(1, len(reviews)))
        self.runtime_stats.logical_claim_aggregate_calls += 1
        self.runtime_stats.logical_input_tokens_estimate += logical_input_tokens
        self.runtime_stats.logical_output_tokens_estimate += 220
        cache_key = stable_hash(
            f"{self.config.prompt_version}|claim_aggregate|{row['claim_object']}|{row['user_claim']}|{row['image_paths']}"
        )
        payload = self._load_cache("claim_review", cache_key)
        if payload is None:
            payload = self._provider_chain(
                "aggregate_claim",
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
                parsed_summary=parsed.claim_summary,
                parsed_issue_type=parsed.issue_type,
                parsed_object_part=parsed.object_part,
                image_reviews=[
                    {
                        "image_id": review.image_id,
                        "claim_status": review.claim_status,
                        "visible_issue_type": review.visible_issue_type,
                        "visible_object_part": review.visible_object_part,
                        "evidence_sufficient": review.evidence_sufficient,
                        "claimed_part_visible": review.claimed_part_visible,
                        "severity": review.severity,
                        "risk_flags": review.risk_flags,
                        "justification": review.justification,
                        "confidence": review.confidence,
                    }
                    for review in reviews
                ],
                evidence_rules=self._evidence_rules(row["claim_object"], parsed.issue_type),
                allowed_parts=sorted(OBJECT_PARTS[row["claim_object"]]),
                allowed_issues=sorted(ISSUE_TYPES),
                allowed_risk_flags=RISK_FLAG_ORDER,
                history_flags=self._history_risk_flags(row["user_id"]),
            )
            if not payload:
                return None
            self._save_cache("claim_review", cache_key, payload)
            self.runtime_stats.input_tokens_estimate += logical_input_tokens
            self.runtime_stats.output_tokens_estimate += 220
        elif not payload:
            return None

        issue_type = normalize_issue_value(str(payload.get("issue_type", parsed.issue_type)), fallback=parsed.issue_type)
        object_part = normalize_object_part_value(
            str(payload.get("object_part", parsed.object_part)),
            row["claim_object"],
            fallback=parsed.object_part,
        )
        history_flags = self._history_risk_flags(row["user_id"])
        risk_flags = [
            flag
            for flag in coerce_list(payload.get("risk_flags", []))
            if flag in RISK_FLAGS and flag != "none"
        ]
        for qc in qcs:
            if not qc.usable and "cropped_or_obstructed" not in risk_flags:
                risk_flags.append("cropped_or_obstructed")
            if qc.edge_variance < 140 and "blurry_image" not in risk_flags:
                risk_flags.append("blurry_image")
            if (qc.brightness < 24 or qc.bright_fraction > 0.4) and "low_light_or_glare" not in risk_flags:
                risk_flags.append("low_light_or_glare")
        for flag in history_flags:
            if flag not in risk_flags:
                risk_flags.append(flag)
        if parsed.prompt_injection_detected and "text_instruction_present" not in risk_flags:
            risk_flags.append("text_instruction_present")

        supported_reviews = [review for review in reviews if review.claim_status == "supported"]
        contradicted_reviews = [review for review in reviews if review.claim_status == "contradicted"]
        strong_supported_reviews = [
            review
            for review in supported_reviews
            if review.evidence_sufficient and review.visible_issue_type != "unknown" and review.visible_object_part != "unknown"
        ]
        strong_contradicted_reviews = [
            review
            for review in contradicted_reviews
            if review.evidence_sufficient or review.visible_issue_type != "unknown" or review.visible_object_part != "unknown"
        ]
        corroborated_non_original = (
            sum(1 for review in reviews if "non_original_image" in review.risk_flags) >= 2
            or any(
                "non_original_image" in review.risk_flags
                and any(flag in review.risk_flags for flag in {"possible_manipulation", "text_instruction_present"})
                for review in reviews
            )
        )
        if strong_supported_reviews and not contradicted_reviews and not corroborated_non_original:
            risk_flags = [flag for flag in risk_flags if flag != "non_original_image"]
        if strong_supported_reviews and any(review.claimed_part_visible for review in strong_supported_reviews):
            risk_flags = [flag for flag in risk_flags if flag != "damage_not_visible"]
        if strong_supported_reviews and not contradicted_reviews:
            risk_flags = [flag for flag in risk_flags if flag != "claim_mismatch"]

        claim_status = normalize_claim_status_value(str(payload.get("claim_status", "not_enough_information")))
        if claim_status == "not_enough_information" and strong_supported_reviews and not any(
            flag in risk_flags
            for flag in {"wrong_object", "claim_mismatch", "possible_manipulation", "text_instruction_present", "non_original_image"}
        ):
            claim_status = "supported"
        if claim_status == "supported" and any(
            flag in risk_flags for flag in {"wrong_object", "claim_mismatch", "possible_manipulation", "text_instruction_present"}
        ):
            claim_status = "not_enough_information"
        if claim_status == "supported" and "non_original_image" in risk_flags and corroborated_non_original:
            claim_status = "not_enough_information"
        if claim_status == "supported" and issue_type == "unknown":
            if strong_supported_reviews:
                issue_type = strong_supported_reviews[0].visible_issue_type
                object_part = strong_supported_reviews[0].visible_object_part
            if issue_type == "unknown":
                claim_status = "not_enough_information"
        evidence_standard_met = coerce_bool(payload.get("evidence_standard_met"), default=claim_status != "not_enough_information")
        if claim_status == "supported" and strong_supported_reviews:
            evidence_standard_met = True
        if claim_status == "contradicted" and strong_contradicted_reviews:
            evidence_standard_met = True
        if claim_status == "not_enough_information" or not any(qc.usable for qc in qcs):
            evidence_standard_met = False

        valid_image = coerce_bool(payload.get("valid_image"), default=any(qc.usable for qc in qcs))
        if not any(qc.usable for qc in qcs):
            valid_image = False
        elif any(flag in risk_flags for flag in {"possible_manipulation", "text_instruction_present"}):
            valid_image = False
        elif "non_original_image" in risk_flags and corroborated_non_original and not strong_supported_reviews:
            valid_image = False
        elif strong_supported_reviews and not corroborated_non_original:
            valid_image = True

        if claim_status == "not_enough_information":
            if not any(review.claimed_part_visible for review in reviews) and "damage_not_visible" not in risk_flags:
                risk_flags.append("damage_not_visible")
        else:
            risk_flags = [flag for flag in risk_flags if flag != "damage_not_visible"]

        manual_review_required = False
        if any(flag in risk_flags for flag in {"wrong_object", "possible_manipulation", "text_instruction_present"}):
            manual_review_required = True
        if "claim_mismatch" in risk_flags and claim_status != "supported":
            manual_review_required = True
        if "non_original_image" in risk_flags and corroborated_non_original:
            manual_review_required = True
        if "user_history_risk" in risk_flags:
            manual_review_required = True
        if strong_supported_reviews and contradicted_reviews:
            manual_review_required = True
        if parsed.mentions_multiple_parts and claim_status == "not_enough_information" and not strong_supported_reviews:
            manual_review_required = True
        if manual_review_required and "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        if not manual_review_required:
            risk_flags = [flag for flag in risk_flags if flag != "manual_review_required"]

        supporting_ids = [
            image_id
            for image_id in coerce_list(payload.get("supporting_image_ids", []))
            if any(qc.image_id == image_id for qc in qcs)
        ]
        if claim_status == "supported" and strong_supported_reviews:
            strong_ids = {review.image_id for review in strong_supported_reviews}
            supporting_ids = [image_id for image_id in supporting_ids if image_id in strong_ids]
            if not supporting_ids:
                best_review = max(strong_supported_reviews, key=lambda review: (review.confidence, review.claimed_part_visible))
                supporting_ids = [best_review.image_id]
        elif claim_status == "contradicted" and strong_contradicted_reviews:
            strong_ids = {review.image_id for review in strong_contradicted_reviews}
            supporting_ids = [image_id for image_id in supporting_ids if image_id in strong_ids]
            if not supporting_ids:
                best_review = max(strong_contradicted_reviews, key=lambda review: (review.confidence, review.evidence_sufficient))
                supporting_ids = [best_review.image_id]
        if claim_status != "not_enough_information" and not supporting_ids and qcs:
            supporting_ids = [qcs[0].image_id]
        supporting_ids = list(dict.fromkeys(supporting_ids))
        supporting_image_ids = ";".join(supporting_ids) if supporting_ids else "none"

        severity = normalize_severity_value(
            str(payload.get("severity", choose_severity(issue_type))),
            issue_type,
            claim_status,
            risk_flags,
        )
        if claim_status == "contradicted" and "wrong_object" in risk_flags and issue_type == "unknown":
            object_part = "unknown"

        evidence_reason = normalize_spaces(
            str(
                payload.get(
                    "evidence_standard_met_reason",
                    "The submitted images are clear enough to evaluate the claim."
                    if evidence_standard_met
                    else "The submitted images do not provide enough reliable evidence to evaluate the claim.",
                )
            )
        )
        status_justification = normalize_spaces(
            str(
                payload.get(
                    "claim_status_justification",
                    "The multimodal reviewer evaluated the full image set against the stated claim.",
                )
            )
        )
        if not evidence_standard_met and claim_status == "not_enough_information" and "not enough" not in lower_alnum(status_justification):
            status_justification = normalize_spaces(
                status_justification + " The image set is not strong enough for a definitive confirmation or contradiction."
            )

        values = {
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            "evidence_standard_met": to_bool_text(evidence_standard_met),
            "evidence_standard_met_reason": evidence_reason,
            "risk_flags": ";".join(flag for flag in RISK_FLAG_ORDER if flag in risk_flags) if risk_flags else "none",
            "issue_type": issue_type if issue_type in ISSUE_TYPES else "unknown",
            "object_part": object_part if object_part in OBJECT_PARTS[row["claim_object"]] else "unknown",
            "claim_status": claim_status,
            "claim_status_justification": status_justification,
            "supporting_image_ids": supporting_image_ids,
            "valid_image": to_bool_text(valid_image),
            "severity": severity if severity in SEVERITIES else "unknown",
        }
        return Prediction(row=row, values=values)

    def _fallback_review(self, row: dict[str, str], parsed: ParsedClaim, qcs: list[ImageQC]) -> list[ImageReview]:
        image_paths = [qc.image_path for qc in qcs if qc.usable]
        best = self.sample_matcher.best_match(row["claim_object"], parsed, image_paths)
        if best is None:
            return [
                ImageReview(
                    image_id=qc.image_id,
                    claim_status="not_enough_information",
                    visible_issue_type="unknown",
                    visible_object_part=parsed.object_part if parsed.object_part != "unknown" else "unknown",
                    evidence_sufficient=False,
                    claimed_part_visible=False,
                    severity="unknown",
                    risk_flags=list(qc.risk_flags),
                    justification="The fallback reviewer could not find a strong exemplar match for this image.",
                    confidence=0.1,
                )
                for qc in qcs
            ]
        exemplar = best["entry"]["row"]
        score = float(best["score"])
        matched_target_index = best.get("matched_target_index")
        best_qc = None
        if isinstance(matched_target_index, int) and 0 <= matched_target_index < len(qcs):
            candidate_qc = qcs[matched_target_index]
            if candidate_qc.usable:
                best_qc = candidate_qc
        if best_qc is None:
            best_qc = max((qc for qc in qcs if qc.usable), key=lambda qc: qc.edge_variance, default=qcs[0] if qcs else None)
        best_image_id = best_qc.image_id if best_qc else "none"
        claim_status = exemplar["claim_status"]
        evidence_sufficient = score >= 0.60 and any(qc.usable for qc in qcs)
        if claim_status == "supported" and score < 0.60:
            claim_status = "not_enough_information"
        if not evidence_sufficient and score < 0.47:
            claim_status = "not_enough_information"
        if score < 0.52 and claim_status == "contradicted":
            claim_status = "not_enough_information"
        issue_type = exemplar["issue_type"] if score >= 0.54 else parsed.issue_type
        object_part = exemplar["object_part"] if score >= 0.54 else parsed.object_part
        justification = (
            f"Fallback exemplar score {score:.2f} most closely matched a labeled {claim_status} sample "
            f"for {exemplar['claim_object']} evidence."
        )
        reviews: list[ImageReview] = []
        for qc in qcs:
            local_status = claim_status if qc.usable else "not_enough_information"
            local_issue = issue_type if qc.image_id == best_image_id or len(qcs) == 1 else parsed.issue_type
            local_part = object_part if qc.image_id == best_image_id or len(qcs) == 1 else parsed.object_part
            risk_flags = list(qc.risk_flags)
            if claim_status == "contradicted" and parsed.issue_type != "unknown" and issue_type not in {"unknown", parsed.issue_type}:
                risk_flags.append("claim_mismatch")
            reviews.append(
                ImageReview(
                    image_id=qc.image_id,
                    claim_status=local_status,
                    visible_issue_type=local_issue if local_issue in ISSUE_TYPES else "unknown",
                    visible_object_part=local_part if local_part in OBJECT_PARTS[row["claim_object"]] else "unknown",
                    evidence_sufficient=evidence_sufficient and qc.image_id == best_image_id,
                    claimed_part_visible=evidence_sufficient,
                    severity=exemplar["severity"] if exemplar["severity"] in SEVERITIES else choose_severity(issue_type),
                    risk_flags=list(dict.fromkeys(flag for flag in risk_flags if flag in RISK_FLAGS and flag != "none")),
                    justification=justification,
                    confidence=min(0.8, max(0.2, score if qc.image_id == best_image_id else score - 0.12)),
                )
            )
        return reviews

    def _aggregate_prediction(
        self,
        row: dict[str, str],
        parsed: ParsedClaim,
        qcs: list[ImageQC],
        reviews: list[ImageReview],
    ) -> Prediction:
        history_flags = self._history_risk_flags(row["user_id"])
        all_usable = any(qc.usable for qc in qcs)
        valid_image = all_usable
        evidence_standard_met = any(review.evidence_sufficient for review in reviews)
        supported = [review for review in reviews if review.claim_status == "supported"]
        contradicted = [review for review in reviews if review.claim_status == "contradicted"]
        insufficient = [review for review in reviews if review.claim_status == "not_enough_information"]

        if supported:
            chosen = max(supported, key=lambda review: (review.confidence, review.evidence_sufficient))
        elif contradicted:
            chosen = max(contradicted, key=lambda review: review.confidence)
        elif insufficient:
            chosen = max(insufficient, key=lambda review: review.confidence)
        else:
            chosen = ImageReview(
                image_id="none",
                claim_status="not_enough_information",
                visible_issue_type="unknown",
                visible_object_part=parsed.object_part if parsed.object_part != "unknown" else "unknown",
                evidence_sufficient=False,
                claimed_part_visible=False,
                severity="unknown",
                risk_flags=[],
                justification="No image evidence was available for review.",
                confidence=0.0,
            )

        risk_flags: list[str] = []
        for qc in qcs:
            for flag in qc.risk_flags:
                if flag not in risk_flags:
                    risk_flags.append(flag)
        for review in reviews:
            for flag in review.risk_flags:
                if flag not in risk_flags:
                    risk_flags.append(flag)
        for flag in history_flags:
            if flag not in risk_flags:
                risk_flags.append(flag)
        if parsed.prompt_injection_detected and "text_instruction_present" not in risk_flags:
            risk_flags.append("text_instruction_present")
        if parsed.mentions_multiple_parts and chosen.claim_status == "not_enough_information" and "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        if any(flag in risk_flags for flag in {"claim_mismatch", "wrong_object", "wrong_object_part", "possible_manipulation", "non_original_image", "text_instruction_present"}):
            if "manual_review_required" not in risk_flags:
                risk_flags.append("manual_review_required")
        if history_flags and "user_history_risk" in history_flags and "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        if chosen.claim_status == "contradicted" and chosen.visible_issue_type == "none" and chosen.claimed_part_visible:
            if "damage_not_visible" not in risk_flags:
                risk_flags.append("damage_not_visible")
            risk_flags = [flag for flag in risk_flags if flag != "claim_mismatch"]
        if chosen.claim_status == "not_enough_information" and chosen.visible_issue_type == "unknown" and "damage_not_visible" not in risk_flags:
            risk_flags.append("damage_not_visible")
        risk_flags = [flag for flag in RISK_FLAG_ORDER if flag in risk_flags]

        if chosen.claim_status == "supported" and not evidence_standard_met:
            chosen = ImageReview(
                image_id=chosen.image_id,
                claim_status="not_enough_information",
                visible_issue_type=chosen.visible_issue_type,
                visible_object_part=chosen.visible_object_part,
                evidence_sufficient=False,
                claimed_part_visible=chosen.claimed_part_visible,
                severity="unknown",
                risk_flags=chosen.risk_flags,
                justification=chosen.justification,
                confidence=chosen.confidence,
            )

        if chosen.claim_status == "supported":
            issue_type = chosen.visible_issue_type if chosen.visible_issue_type != "unknown" else parsed.issue_type
            object_part = chosen.visible_object_part if chosen.visible_object_part != "unknown" else parsed.object_part
        elif chosen.claim_status == "contradicted":
            issue_type = chosen.visible_issue_type if chosen.visible_issue_type != "unknown" else "none"
            object_part = chosen.visible_object_part if chosen.visible_object_part != "unknown" else parsed.object_part
        else:
            issue_type = "unknown" if parsed.issue_type == "unknown" else parsed.issue_type
            object_part = parsed.object_part if parsed.object_part != "unknown" else chosen.visible_object_part

        issue_type = issue_type if issue_type in ISSUE_TYPES else "unknown"
        object_part = object_part if object_part in OBJECT_PARTS[row["claim_object"]] else "unknown"
        severity = chosen.severity if chosen.severity in SEVERITIES else choose_severity(issue_type)

        if chosen.claim_status == "contradicted" and issue_type == "none":
            severity = "none"
        if chosen.claim_status == "not_enough_information":
            severity = "unknown"
        if any(flag in risk_flags for flag in {"non_original_image", "possible_manipulation"}) and chosen.claim_status == "contradicted":
            valid_image = False

        if chosen.claim_status == "supported":
            supporting_ids = [
                review.image_id
                for review in reviews
                if review.claim_status == "supported" and review.evidence_sufficient
            ]
            if not supporting_ids:
                supporting_ids = [
                    review.image_id
                    for review in reviews
                    if review.claim_status == "supported" and review.claim_status != "not_enough_information"
                ]
        else:
            supporting_ids = [
                review.image_id
                for review in reviews
                if review.claim_status == chosen.claim_status and review.claim_status != "not_enough_information"
            ]
        supporting_ids = list(dict.fromkeys(item for item in supporting_ids if item != "none"))
        if not supporting_ids and chosen.claim_status != "not_enough_information":
            supporting_ids = [chosen.image_id] if chosen.image_id != "none" else []
        supporting_image_ids = ";".join(supporting_ids) if supporting_ids else "none"

        if evidence_standard_met:
            evidence_reason = (
                f"The claimed {object_part.replace('_', ' ')} is visible clearly enough in the submitted image set to evaluate the claim."
            )
        else:
            evidence_reason = (
                f"The submitted images do not show the claimed {parsed.object_part.replace('_', ' ')} clearly enough for a reliable decision."
                if parsed.object_part != "unknown"
                else "The submitted images do not provide enough clear evidence to evaluate the claim."
            )

        if chosen.claim_status == "supported":
            status_justification = f"{chosen.justification} Supporting images: {supporting_image_ids}."
        elif chosen.claim_status == "contradicted":
            status_justification = f"{chosen.justification} The available images support a contradiction rather than the stated claim."
        else:
            status_justification = f"{chosen.justification} The image set is not strong enough for a definitive confirmation or contradiction."

        values = {
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            "evidence_standard_met": to_bool_text(evidence_standard_met),
            "evidence_standard_met_reason": normalize_spaces(evidence_reason),
            "risk_flags": ";".join(risk_flags) if risk_flags else "none",
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": chosen.claim_status,
            "claim_status_justification": normalize_spaces(status_justification),
            "supporting_image_ids": supporting_image_ids,
            "valid_image": to_bool_text(valid_image),
            "severity": severity,
        }
        return Prediction(row=row, values=values)

    def predict_row(self, row: dict[str, str], strategy: str = "hybrid") -> Prediction:
        parsed = parse_claim_heuristic(row["user_claim"], row["claim_object"])

        relative_paths = [item.strip() for item in row["image_paths"].split(";") if item.strip()]
        qcs = []
        for relative_path in relative_paths:
            image_path = resolve_dataset_path(self.config.repo_root, relative_path)
            qc = analyze_image_qc(image_path, image_id_from_path(relative_path))
            qcs.append(qc)
        self.runtime_stats.images_processed += len(qcs)
        live_parsed = self._provider_normalize_claim(row, parsed) if self.providers and strategy in {"hybrid", "ensemble"} else parsed

        if strategy == "text_baseline":
            reviews = [
                ImageReview(
                    image_id=qc.image_id,
                    claim_status="supported" if qc.usable and parsed.issue_type != "unknown" else "not_enough_information",
                    visible_issue_type=parsed.issue_type if parsed.issue_type != "unknown" else "unknown",
                    visible_object_part=parsed.object_part if parsed.object_part != "unknown" else "unknown",
                    evidence_sufficient=qc.usable and not qc.risk_flags,
                    claimed_part_visible=qc.usable,
                    severity=choose_severity(parsed.issue_type),
                    risk_flags=list(qc.risk_flags),
                    justification="The baseline strategy relies on transcript parsing with light image quality checks.",
                    confidence=0.25 if parsed.issue_type != "unknown" else 0.1,
                )
                for qc in qcs
            ]
        elif strategy == "hybrid" and self.providers:
            reviews = [self._provider_review_image(row, live_parsed, qc) for qc in qcs]
            prediction = self._provider_predict_row(row, live_parsed, qcs, reviews)
            if prediction is not None:
                return prediction
            prediction = self._aggregate_live_reviews_fallback(row, live_parsed, qcs, reviews)
            if prediction is not None:
                return prediction
            reviews = self._fallback_review(row, parsed, qcs)
        elif strategy == "ensemble":
            retrieval_reviews = self._fallback_review(row, parsed, qcs)
            retrieval_prediction = self._aggregate_prediction(row, parsed, qcs, retrieval_reviews)
            if not self.providers:
                return retrieval_prediction
            reviews = [self._provider_review_image(row, live_parsed, qc) for qc in qcs]
            prediction = self._provider_predict_row(row, live_parsed, qcs, reviews)
            if prediction is None:
                prediction = self._aggregate_live_reviews_fallback(row, live_parsed, qcs, reviews)
            if prediction is None:
                return retrieval_prediction
            chosen_prediction = self._choose_ensemble_prediction(retrieval_prediction, prediction)
            return self._refine_ensemble_prediction(chosen_prediction, reviews)
        elif strategy == "hybrid":
            reviews = self._fallback_review(row, parsed, qcs)
        else:
            reviews = self._fallback_review(row, parsed, qcs)
        return self._aggregate_prediction(row, parsed, qcs, reviews)

    def predict_rows(self, rows: list[dict[str, str]], strategy: str = "hybrid") -> list[Prediction]:
        return [self.predict_row(row, strategy=strategy) for row in rows]


def evaluate_predictions(
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
    strategy_name: str,
    notes: str = "",
) -> StrategyMetrics:
    comparable_fields = [
        "evidence_standard_met",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]
    matches = Counter()
    exact = 0
    for expected, predicted in zip(expected_rows, predicted_rows):
        row_ok = True
        for field in comparable_fields:
            if normalize_spaces(expected[field]) == normalize_spaces(predicted[field]):
                matches[field] += 1
            else:
                row_ok = False
        if row_ok:
            exact += 1
    row_count = len(expected_rows)
    field_accuracies = {field: matches[field] / row_count for field in comparable_fields}
    return StrategyMetrics(
        name=strategy_name,
        exact_match_accuracy=exact / row_count,
        field_accuracies=field_accuracies,
        row_count=row_count,
        notes=notes,
    )


def build_operational_notes(
    strategy_stats: RuntimeStats,
    total_rows: int,
    avg_images_per_row: float,
    strategy_name: str = "retrieval",
    *,
    live_models_enabled: bool = False,
    evaluation_stats: RuntimeStats | None = None,
) -> dict[str, Any]:
    del total_rows, avg_images_per_row
    evaluation_stats = evaluation_stats or strategy_stats
    selected_strategy_live = strategy_name in {"hybrid", "ensemble"} and live_models_enabled and strategy_stats.logical_provider_calls > 0
    evaluation_used_live = live_models_enabled and (
        evaluation_stats.logical_provider_calls > 0 or evaluation_stats.provider_calls > 0 or evaluation_stats.cache_hits > 0
    )
    actual_provider_requests = evaluation_stats.provider_calls if evaluation_used_live else 0
    cache_hits = evaluation_stats.cache_hits if evaluation_used_live else 0
    if selected_strategy_live:
        model_calls = strategy_stats.logical_provider_calls
        token_in = strategy_stats.logical_input_tokens_estimate or model_calls * 900
        token_out = strategy_stats.logical_output_tokens_estimate or model_calls * 250
        if actual_provider_requests > 0:
            cost_assumption = (
                "Live provider keys were available and this evaluation run issued uncached provider requests. "
                "Cache reuse may also have reduced repeated calls for already-warmed steps."
            )
        else:
            cost_assumption = (
                "Live provider keys were available, but this evaluation run was fully served from cached live responses "
                "and issued no uncached provider requests."
            )
        if strategy_name == "ensemble":
            latency_note = (
                "Ensemble mode runs the offline retrieval fallback first, then a live claim-normalization call, one live image-review call per image, "
                "and one live text-only aggregation call per row before promoting the live answer only when it is cleaner or better grounded."
            )
        else:
            latency_note = "Live mode performs one claim-normalization call, one image-review call per image, and one text-only aggregation call per row. Oversized images are compressed for inline transport and retried with provider/model fallbacks when needed."
        if cache_hits > 0:
            latency_note += " This run also reused cached live responses for already-computed steps."
    elif evaluation_used_live:
        model_calls = 0
        token_in = 0
        token_out = 0
        cost_assumption = (
            "The selected strategy is offline retrieval, but live provider keys were available and this evaluation run still benchmarked live alternatives."
        )
        latency_note = (
            "The selected strategy executes without external model calls. Live strategies were also evaluated during this run for comparison, "
            "with cache reuse reducing repeated provider requests where possible."
        )
    else:
        model_calls = 0
        token_in = 0
        token_out = 0
        cost_assumption = "Live provider keys were not available in the local environment, so the offline retrieval fallback was executed at zero API cost."
        latency_note = "Offline retrieval mode compares local images against the sample-set exemplars and applies deterministic rule arbitration without external model calls."
    return {
        "approx_model_calls": model_calls,
        "approx_input_tokens": token_in,
        "approx_output_tokens": token_out,
        "actual_uncached_provider_requests": actual_provider_requests,
        "cache_hits": cache_hits,
        "images_processed": strategy_stats.images_processed,
        "cost_assumption": cost_assumption,
        "latency_note": latency_note,
        "rate_limit_note": "The pipeline is sequential by default, cache-aware, and retries across current NIM-compatible multimodal models before dropping to the offline retrieval fallback.",
    }
