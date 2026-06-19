"""Static constants and lightweight vocabularies for claim review."""

from __future__ import annotations

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}
ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}
SEVERITIES = {"none", "low", "medium", "high", "unknown"}
RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}
RISK_FLAG_ORDER = [
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

OBJECT_PARTS = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}

ISSUE_KEYWORDS = {
    "glass_shatter": ["shattered", "shatter", "glass shattered", "front glass shattered"],
    "crack": ["crack", "cracked", "spreading crack", "hairline crack"],
    "broken_part": [
        "broken",
        "broke",
        "not sitting",
        "wobbles",
        "wobble",
        "snapped",
        "damaged",
        "not opening smoothly",
    ],
    "missing_part": ["missing", "not inside", "gone", "absent", "did not find", "not there"],
    "torn_packaging": ["torn", "ripped", "phati", "opened", "open box", "seal broken", "phati hui"],
    "crushed_packaging": ["crushed", "crushed in", "collapsed", "crumpled"],
    "water_damage": ["water", "wet", "soaked", "spill", "spilled"],
    "stain": ["stain", "stained", "sticky", "mark from liquid"],
    "dent": ["dent", "dented", "deep dent", "bump"],
    "scratch": ["scratch", "scrape", "scraped", "mark", "scuff"],
}

PART_KEYWORDS = {
    "car": {
        "front_bumper": ["front bumper", "bumper ke upar", "front side bumper", "front bumper area"],
        "rear_bumper": ["rear bumper", "back bumper", "rear side", "back looks", "back bumper area"],
        "door": ["door", "door panel", "side door"],
        "hood": ["hood", "top panel", "bonnet"],
        "windshield": ["windshield", "front glass", "glass", "windscreen"],
        "side_mirror": ["side mirror", "mirror"],
        "headlight": ["headlight", "front light"],
        "taillight": ["taillight", "rear light", "tail light"],
        "fender": ["fender"],
        "quarter_panel": ["quarter panel"],
        "body": ["body", "side panel", "panel area"],
    },
    "laptop": {
        "screen": ["screen", "display", "display glass"],
        "keyboard": ["keyboard", "keys", "key area"],
        "trackpad": ["trackpad", "touchpad"],
        "hinge": ["hinge", "hinge area"],
        "lid": ["lid", "cover"],
        "corner": ["corner", "edge corner"],
        "port": ["port", "charging port", "usb port"],
        "base": ["base", "bottom"],
        "body": ["body", "outer body", "outer shell"],
    },
    "package": {
        "box": ["box", "outside box", "delivery box", "shipping box"],
        "package_corner": ["corner", "box corner", "package corner"],
        "package_side": ["side", "surface", "outside", "package side"],
        "seal": ["seal", "tape", "seal area", "seal wali side"],
        "label": ["label", "shipping label"],
        "contents": ["contents", "inside", "inside item", "product inside"],
        "item": ["item", "product"],
    },
}

PROMPT_INJECTION_PATTERNS = [
    "approve immediately",
    "skip manual review",
    "system reading this",
    "ignore previous",
    "always approve",
    "do not review",
]

DEFAULT_SEVERITY_BY_ISSUE = {
    "none": "none",
    "unknown": "unknown",
    "scratch": "low",
    "stain": "low",
    "dent": "medium",
    "crack": "medium",
    "glass_shatter": "high",
    "broken_part": "medium",
    "missing_part": "high",
    "torn_packaging": "medium",
    "crushed_packaging": "medium",
    "water_damage": "medium",
}

MODEL_DEFAULTS = {
    "nim": "stepfun-ai/step-3.7-flash",
    "openrouter": "qwen/qwen2.5-vl-72b-instruct",
}
