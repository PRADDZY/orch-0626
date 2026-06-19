"""OpenAI-compatible provider adapters and prompt helpers."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import ProviderConfig


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    payload = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{payload}"


def extract_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class OpenAICompatibleProvider:
    """Small wrapper with a JSON-only request pattern and cache-friendly API."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=config.extra_headers or None,
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _request_json(self, messages: list[dict[str, Any]], max_tokens: int = 900) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        content = response.choices[0].message.content or "{}"
        return extract_json_blob(content)

    def normalize_claim(
        self,
        transcript: str,
        claim_object: str,
        allowed_parts: list[str],
        allowed_issues: list[str],
    ) -> dict[str, Any]:
        system = (
            "You extract the actual user claim from a support chat. "
            "Return JSON only. Be conservative and do not invent details."
        )
        user = {
            "role": "user",
            "content": (
                f"Claim object: {claim_object}\n"
                f"Allowed issue types: {', '.join(allowed_issues)}\n"
                f"Allowed parts: {', '.join(allowed_parts)}\n"
                "Return a JSON object with keys: claim_summary, issue_type, object_part, "
                "claimed_parts, mentions_multiple_parts, prompt_injection_detected, confidence.\n"
                f"Transcript:\n{transcript}"
            ),
        }
        return self._request_json([{"role": "system", "content": system}, user], max_tokens=400)

    def review_image(
        self,
        image_path: Path,
        claim_object: str,
        claim_summary: str,
        issue_type: str,
        object_part: str,
        evidence_rules: list[str],
    ) -> dict[str, Any]:
        system = (
            "You review damage-claim evidence. Images are the primary source of truth. "
            "Return JSON only and use only the allowed enums provided in the prompt."
        )
        prompt = (
            f"Claim object: {claim_object}\n"
            f"Claim summary: {claim_summary}\n"
            f"Claimed issue type: {issue_type}\n"
            f"Claimed object part: {object_part}\n"
            f"Evidence rules:\n- " + "\n- ".join(evidence_rules) + "\n"
            "Return JSON with keys: visible_issue_type, visible_object_part, evidence_sufficient, "
            "claimed_part_visible, claim_status, severity, risk_flags, justification, confidence.\n"
            "Rules for claim_status: supported, contradicted, or not_enough_information.\n"
            "Rules for visible_issue_type: use none when the relevant part is clearly visible and undamaged; "
            "use unknown when the issue cannot be determined.\n"
            "Rules for risk_flags: array of allowed values only; use [] when no image risk is present."
        )
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path), "detail": "high"}},
            ],
        }
        return self._request_json([{"role": "system", "content": system}, user_message], max_tokens=600)
