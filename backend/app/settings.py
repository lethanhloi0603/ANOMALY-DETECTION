"""Application settings loaded from environment variables and JSON config.

The persistence layer deliberately avoids an additional settings dependency.  This
keeps the development SQLite setup zero-dependency while allowing production to
provide a PostgreSQL URL and a versioned framework catalog.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent


def _as_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _as_int(value: str | int | None, default: int, *, minimum: int = 0) -> int:
    if value is None:
        result = default
    else:
        result = int(value)
    if result < minimum:
        raise ValueError(f"Expected an integer >= {minimum}, got {result}")
    return result


def _as_float(
    value: str | float | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    result = default if value is None else float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"Expected a number >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"Expected a number <= {maximum}, got {result}")
    return result


def _as_optional_float(
    value: str | float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _as_float(value, 0.0, minimum=minimum, maximum=maximum)


def _as_origins(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


def _optional_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _principal(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings.

    ``framework_config_path`` points to the JSON file used by ``seed_catalog``.
    Missing JSON is tolerated so database initialization also works before a
    project-specific catalog is authored.
    """

    database_url: str
    app_env: str
    app_host: str
    app_port: int
    log_level: str
    api_key: str | None
    api_key_actor: str
    scorer_api_key: str | None
    scorer_api_key_actor: str
    cors_origins: tuple[str, ...]
    max_batch_events: int
    max_request_bytes: int
    locked_alert_threshold: float | None
    auto_create_schema: bool
    framework_config_path: Path
    feature_catalog_path: Path
    sequence_config_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        configured_path = (
            os.getenv("FRAMEWORK_CONFIG_PATH")
            or os.getenv("CONFIG_PATH")
            or str(BACKEND_ROOT / "config" / "framework.v4.json")
        )
        feature_catalog_path = os.getenv("FEATURE_CATALOG_PATH") or str(
            REPOSITORY_ROOT / "machine_learning" / "config" / "feature128.v5.json"
        )
        sequence_config_path = os.getenv("SEQUENCE_CONFIG_PATH") or str(
            REPOSITORY_ROOT / "machine_learning" / "config" / "sequence7.v4.json"
        )
        default_database_path = (
            REPOSITORY_ROOT / "data" / "runtime" / "insider_threat.db"
        ).as_posix()
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                f"sqlite:///{default_database_path}",
            ),
            app_env=os.getenv("APP_ENV", "development"),
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=_as_int(os.getenv("APP_PORT"), 8000, minimum=1),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            api_key=_optional_secret("API_KEY"),
            api_key_actor=_principal("API_KEY_ACTOR", "api-key-principal"),
            scorer_api_key=_optional_secret("SCORER_API_KEY"),
            scorer_api_key_actor=_principal(
                "SCORER_API_KEY_ACTOR",
                "trusted-scorer",
            ),
            cors_origins=_as_origins(os.getenv("CORS_ORIGINS")),
            max_batch_events=_as_int(
                os.getenv("MAX_BATCH_EVENTS"),
                1000,
                minimum=1,
            ),
            max_request_bytes=_as_int(
                os.getenv("MAX_REQUEST_BYTES"),
                2_097_152,
                minimum=1,
            ),
            locked_alert_threshold=_as_optional_float(
                os.getenv("LOCKED_ALERT_THRESHOLD"),
                minimum=0.0,
                maximum=1.0,
            ),
            auto_create_schema=_as_bool(
                os.getenv("AUTO_CREATE_SCHEMA"),
                True,
            ),
            framework_config_path=Path(configured_path).expanduser().resolve(),
            feature_catalog_path=Path(feature_catalog_path).expanduser().resolve(),
            sequence_config_path=Path(sequence_config_path).expanduser().resolve(),
        )

    def load_framework_config(self, *, required: bool = False) -> dict[str, Any]:
        """Read and validate the configured JSON object.

        The loader returns an empty mapping when the default file is absent.  Set
        ``required=True`` in deployment/bootstrap tooling when a catalog is a
        mandatory release artifact.
        """

        path = self.framework_config_path
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Framework config does not exist: {path}")
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in framework config {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Framework config root must be a JSON object: {path}")
        return payload

    def load_feature_catalog(self, *, required: bool = False) -> dict[str, Any]:
        path = self.feature_catalog_path
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Feature catalog does not exist: {path}")
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in feature catalog {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Feature catalog root must be a JSON object: {path}")
        return payload

    def load_sequence_config(self, *, required: bool = False) -> dict[str, Any]:
        path = self.sequence_config_path
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Sequence config does not exist: {path}")
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in sequence config {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Sequence config root must be a JSON object: {path}")
        return payload


settings = Settings.from_env()
