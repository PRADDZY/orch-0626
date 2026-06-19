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
    multiple_parts = len(part_hits) > 1 or "two things" in text or "multiple parts" in text
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
            if edge_variance < 120:
                risk_flags.append("blurry_image")
            if brightness < 40 or bright_fraction > 0.28:
                risk_flags.append("low_light_or_glare")
            if min(width, height) < 180:
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

    def score(self, target_features: list[dict[str, Any]], exemplar: dict[str, Any], parsed: ParsedClaim) -> float:
        if not target_features or not exemplar["features"]:
            image_score = 0.0
        else:
            image_score = 0.0
            for target in target_features:
                for candidate in exemplar["features"]:
                    score = (
                        hamming_similarity(target["ahash"], candidate["ahash"]) * 0.45
                        + hamming_similarity(target["dhash"], candidate["dhash"]) * 0.35
                        + color_similarity(target["color"], candidate["color"]) * 0.20
                    )
                    image_score = max(image_score, score)
        issue_bonus = 0.18 if parsed.issue_type != "unknown" and parsed.issue_type == exemplar["row"]["issue_type"] else 0.0
        part_bonus = 0.14 if parsed.object_part != "unknown" and parsed.object_part == exemplar["row"]["object_part"] else 0.0
        return image_score * 0.68 + issue_bonus + part_bonus

    def best_match(
        self,
        claim_object: str,
        parsed: ParsedClaim,
        image_paths: list[Path],
    ) -> dict[str, Any] | None:
        target_features = []
        for image_path in image_paths:
            try:
                target_features.append(self._image_features(image_path))
            except Exception:
                continue
        best: dict[str, Any] | None = None
        best_score = -1.0
        for entry in self.entries:
            if entry["claim_object"] != claim_object:
                continue
            score = self.score(target_features, entry, parsed)
            if score > best_score:
                best = {"entry": entry, "score": score}
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

    def _provider_normalize_claim(self, row: dict[str, str], parsed: ParsedClaim) -> ParsedClaim:
        if not self.providers:
            return parsed
        allowed_parts = sorted(OBJECT_PARTS[row["claim_object"]])
        allowed_issues = sorted(ISSUE_TYPES)
        cache_key = stable_hash(f"{self.config.prompt_version}|normalize|{row['claim_object']}|{row['user_claim']}")
        cached = self._load_cache("normalize", cache_key)
        payload = cached
        if payload is None:
            provider = self.providers[0]
            payload = provider.normalize_claim(row["user_claim"], row["claim_object"], allowed_parts, allowed_issues)
            self._save_cache("normalize", cache_key, payload)
            self.runtime_stats.provider_calls += 1
            self.runtime_stats.input_tokens_estimate += max(80, len(row["user_claim"]) // 3)
            self.runtime_stats.output_tokens_estimate += 120
        issue_type = payload.get("issue_type", parsed.issue_type)
        if issue_type not in ISSUE_TYPES:
            issue_type = parsed.issue_type
        object_part = payload.get("object_part", parsed.object_part)
        if object_part not in OBJECT_PARTS[row["claim_object"]]:
            object_part = parsed.object_part
        claimed_parts = [part for part in payload.get("claimed_parts", parsed.claimed_parts) if part in OBJECT_PARTS[row["claim_object"]]]
        return ParsedClaim(
            claim_summary=normalize_spaces(payload.get("claim_summary", parsed.claim_summary)),
            issue_type=issue_type,
            object_part=object_part,
            claimed_parts=claimed_parts or parsed.claimed_parts,
            mentions_multiple_parts=bool(payload.get("mentions_multiple_parts", parsed.mentions_multiple_parts)),
            prompt_injection_detected=bool(payload.get("prompt_injection_detected", parsed.prompt_injection_detected)),
            confidence=float(payload.get("confidence", parsed.confidence or 0.0)),
        )

    def _provider_review_image(
        self,
        row: dict[str, str],
        parsed: ParsedClaim,
        qc: ImageQC,
    ) -> ImageReview:
        if not self.providers:
            raise RuntimeError("provider review requested without any providers")
        cache_key = stable_hash(
            f"{self.config.prompt_version}|image|{row['claim_object']}|{parsed.claim_summary}|{qc.image_path}|{qc.edge_variance:.1f}"
        )
        cached = self._load_cache("image_review", cache_key)
        payload = cached
        if payload is None:
            provider = self.providers[0]
            payload = provider.review_image(
                image_path=qc.image_path,
                claim_object=row["claim_object"],
                claim_summary=parsed.claim_summary,
                issue_type=parsed.issue_type,
                object_part=parsed.object_part,
                evidence_rules=self._evidence_rules(row["claim_object"], parsed.issue_type),
            )
            self._save_cache("image_review", cache_key, payload)
            self.runtime_stats.provider_calls += 1
            self.runtime_stats.input_tokens_estimate += 200
            self.runtime_stats.output_tokens_estimate += 180
        risk_flags = [flag for flag in payload.get("risk_flags", []) if flag in RISK_FLAGS and flag != "none"]
        risk_flags.extend(flag for flag in qc.risk_flags if flag not in risk_flags)
        claim_status = payload.get("claim_status", "not_enough_information")
        if claim_status not in CLAIM_STATUSES:
            claim_status = "not_enough_information"
        issue_type = payload.get("visible_issue_type", "unknown")
        if issue_type not in ISSUE_TYPES:
            issue_type = "unknown"
        object_part = payload.get("visible_object_part", "unknown")
        if object_part not in OBJECT_PARTS[row["claim_object"]]:
            object_part = "unknown"
        severity = payload.get("severity", choose_severity(issue_type))
        if severity not in SEVERITIES:
            severity = choose_severity(issue_type)
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
            confidence=float(payload.get("confidence", 0.5)),
        )

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
        if parsed.mentions_multiple_parts and "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        if any(flag in risk_flags for flag in {"claim_mismatch", "wrong_object", "wrong_object_part", "possible_manipulation", "non_original_image", "text_instruction_present"}):
            if "manual_review_required" not in risk_flags:
                risk_flags.append("manual_review_required")
        if history_flags and "user_history_risk" in history_flags and "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
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
        if strategy == "hybrid" and self.providers:
            parsed = self._provider_normalize_claim(row, parsed)

        relative_paths = [item.strip() for item in row["image_paths"].split(";") if item.strip()]
        qcs = []
        for relative_path in relative_paths:
            image_path = resolve_dataset_path(self.config.repo_root, relative_path)
            qc = analyze_image_qc(image_path, image_id_from_path(relative_path))
            qcs.append(qc)
        self.runtime_stats.images_processed += len(qcs)

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
            reviews = [self._provider_review_image(row, parsed, qc) for qc in qcs]
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


def build_operational_notes(reviewer: ClaimReviewer, total_rows: int, avg_images_per_row: float) -> dict[str, Any]:
    live_path = reviewer.config.enable_live_models
    if live_path:
        model_calls = total_rows * (1 + math.ceil(avg_images_per_row))
        token_in = model_calls * 220
        token_out = model_calls * 180
        cost_assumption = "NVIDIA Build developer endpoint assumed to be free during hackathon development; OpenRouter fallback cost not incurred unless explicitly enabled."
    else:
        model_calls = 0
        token_in = 0
        token_out = 0
        cost_assumption = "Live provider keys were not available in the local environment, so the offline retrieval fallback was executed at zero API cost."
    return {
        "approx_model_calls": model_calls,
        "approx_input_tokens": token_in,
        "approx_output_tokens": token_out,
        "images_processed": reviewer.runtime_stats.images_processed,
        "cost_assumption": cost_assumption,
        "latency_note": "Live mode is designed for one normalization call plus one image-review call per image, with file-backed caching to avoid repeats.",
        "rate_limit_note": "The pipeline is sequential by default, cache-aware, and can be batched later if provider RPM limits become visible during a live run.",
    }
