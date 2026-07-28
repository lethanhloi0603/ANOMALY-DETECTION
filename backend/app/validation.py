from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any, TypeVar

LABEL_KEY_PATTERN = re.compile(
    r"(^|_)(label|labels|scenario|insider|insider_flag|malicious|ground_truth|answer|"
    r"answer_key|is_threat)($|_)",
    re.IGNORECASE,
)
CONTENT_KEY_PATTERN = re.compile(
    r"(^|_)(content|content_keywords|keywords|message_body|body_text|ocean)($|_)",
    re.IGNORECASE,
)

MINUTE_FEATURES = {"first_logon_minute", "last_logon_minute", "last_logoff_minute"}
T = TypeVar("T")


class DomainValidationError(ValueError):
    def __init__(self, code: str, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def iter_nested_keys(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from iter_nested_keys(nested, path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from iter_nested_keys(nested, f"{prefix}[{index}]")


def enforce_label_firewall(payload: Mapping[str, Any]) -> None:
    violations: list[str] = []
    for path in iter_nested_keys(payload):
        leaf = re.split(r"[.\[]", path)[-1].rstrip("]")
        normalized = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
        if LABEL_KEY_PATTERN.search(normalized):
            violations.append(path)
    if violations:
        raise DomainValidationError(
            "LABEL_FIREWALL_VIOLATION",
            "model-input payload contains evaluation-label fields",
            {"paths": sorted(set(violations))},
        )


def enforce_metadata_only_payload(payload: Mapping[str, Any]) -> None:
    violations: list[str] = []
    for path in iter_nested_keys(payload):
        leaf = re.split(r"[.\[]", path)[-1].rstrip("]")
        normalized = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
        if CONTENT_KEY_PATTERN.search(normalized):
            violations.append(path)
    if violations:
        raise DomainValidationError(
            "PRIMARY_METADATA_ONLY_VIOLATION",
            "primary model payload cannot retain content keywords, message body, or OCEAN",
            {"paths": sorted(set(violations))},
        )


def enforce_cert_http_payload(payload: Mapping[str, Any]) -> None:
    unsupported = {
        "download_bytes",
        "http_method",
        "ip",
        "method",
        "request_bytes",
        "response_bytes",
        "response_status",
        "status",
        "status_code",
        "upload_bytes",
    }
    violations: list[str] = []
    for path in iter_nested_keys(payload):
        leaf = re.split(r"[.\[]", path)[-1].rstrip("]")
        normalized = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
        if normalized in unsupported:
            violations.append(path)
    if violations:
        raise DomainValidationError(
            "CERT_HTTP_FIELD_UNSUPPORTED",
            "CERT HTTP rows do not provide method/status/IP/upload/download byte fields",
            {"paths": sorted(set(violations))},
        )


def validate_feature_vector(
    values: Mapping[str, float | None],
    masks: Mapping[str, bool],
    expected_names: set[str],
) -> None:
    actual_names = set(values)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise DomainValidationError(
            "FEATURE_SCHEMA_MISMATCH",
            "feature vector must contain exactly the locked feature schema",
            {
                "expected": len(expected_names),
                "actual": len(actual_names),
                "missing": missing,
                "extra": extra,
            },
        )

    unknown_masks = sorted(set(masks) - expected_names)
    if unknown_masks:
        raise DomainValidationError(
            "FEATURE_MASK_UNKNOWN",
            "feature masks contain unknown feature names",
            {"features": unknown_masks},
        )

    errors: dict[str, str] = {}
    for name, value in values.items():
        present = masks.get(name, value is not None)
        if value is None:
            if present:
                errors[name] = "undefined feature must have mask=false"
            continue
        if not math.isfinite(value):
            errors[name] = "must be finite"
        elif value < 0:
            errors[name] = "raw feature values must be non-negative"
        elif name.endswith("_ratio") and value > 1:
            errors[name] = "ratio must be within [0,1]"
        elif name.endswith("_entropy") and value < 0:
            errors[name] = "entropy must be >= 0"
        elif name in MINUTE_FEATURES and value > 1439:
            errors[name] = "minute-of-day must be within [0,1439]"
    if errors:
        raise DomainValidationError(
            "FEATURE_VALUE_INVALID",
            "feature vector contains invalid values",
            errors,
        )


def truncate_sequence(values: list[T], max_len: int = 256) -> tuple[list[T], bool]:
    if len(values) <= max_len:
        return values, False
    half = max_len // 2
    return [*values[:half], *values[-(max_len - half) :]], True


def dataset_split(day: date) -> str:
    if date(2010, 1, 2) <= day <= date(2010, 5, 31):
        return "TRAIN"
    if date(2010, 6, 1) <= day <= date(2010, 9, 30):
        return "VALIDATION"
    if date(2010, 10, 1) <= day <= date(2011, 5, 17):
        return "TEST"
    return "PRODUCTION"


def risk_severity(risk: float) -> str:
    if risk >= 0.99:
        return "CRITICAL"
    if risk >= 0.97:
        return "HIGH"
    if risk >= 0.95:
        return "MEDIUM"
    return "LOW"
