from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.settings import settings

FEATURE_SCHEMA_VERSION = "feature128.v5"
FRAMEWORK_SCHEMA_VERSION = "framework.v5"
SEQUENCE_SCHEMA_VERSION = "sequence7.v4"


@lru_cache(maxsize=1)
def load_feature_catalog() -> dict[str, Any]:
    with settings.feature_catalog_path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    features = catalog.get("features", [])
    if catalog.get("schema_version") != FEATURE_SCHEMA_VERSION or len(features) != 128:
        raise RuntimeError(f"locked {FEATURE_SCHEMA_VERSION} catalog is missing or invalid")
    ids = {feature["id"] for feature in features}
    names = {feature["name"] for feature in features}
    if len(ids) != 128 or len(names) != 128:
        raise RuntimeError(f"{FEATURE_SCHEMA_VERSION} IDs and names must be unique")
    return catalog


@lru_cache(maxsize=1)
def load_framework_config() -> dict[str, Any]:
    with settings.framework_config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != FRAMEWORK_SCHEMA_VERSION:
        raise RuntimeError(f"{FRAMEWORK_SCHEMA_VERSION} configuration is missing or invalid")
    return config


@lru_cache(maxsize=1)
def load_sequence_config() -> dict[str, Any]:
    with settings.sequence_config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    tokens = config.get("tokens", [])
    if config.get("schema_version") != SEQUENCE_SCHEMA_VERSION or len(tokens) != 7:
        raise RuntimeError(f"{SEQUENCE_SCHEMA_VERSION} configuration is missing or invalid")
    return config


def feature_names() -> set[str]:
    return {feature["name"] for feature in load_feature_catalog()["features"]}
