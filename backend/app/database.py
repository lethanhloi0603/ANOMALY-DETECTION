"""SQLAlchemy engine, sessions, schema bootstrap, and catalog seeding."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, MetaData, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.settings import settings

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(
    dbapi_connection: object,
    _connection_record: object,
) -> None:
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def make_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    """Create a portable SQLite/PostgreSQL engine.

    An in-memory SQLite URL receives a ``StaticPool`` so the schema remains
    visible across sessions and FastAPI dependency calls during tests.
    """

    url = database_url or settings.database_url
    kwargs: dict[str, Any] = {"echo": echo}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
    return create_engine(url, **kwargs)


engine = make_engine()
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI-compatible session dependency."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope(bind: Engine | None = None) -> Iterator[Session]:
    """Transactional session helper for workers and maintenance commands."""

    factory = (
        SessionLocal
        if bind is None
        else sessionmaker(
            bind=bind,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )
    )
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _load_json_config(
    config: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if config is None:
        return settings.load_feature_catalog()
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid catalog JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Catalog JSON root must be an object: {path}")
    return payload


def _flatten_feature_rows(raw_features: object) -> list[dict[str, Any]]:
    """Accept a flat feature list or a mapping of group -> feature list."""

    rows: list[dict[str, Any]] = []
    if raw_features is None:
        return rows
    if isinstance(raw_features, Mapping):
        for group_name, group_features in raw_features.items():
            if not isinstance(group_features, Sequence) or isinstance(
                group_features,
                (str, bytes),
            ):
                continue
            for value in group_features:
                row = dict(value) if isinstance(value, Mapping) else {"name": str(value)}
                row.setdefault("group", str(group_name))
                rows.append(row)
        return rows
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raise ValueError("Feature definitions must be a list or group mapping")
    for value in raw_features:
        rows.append(dict(value) if isinstance(value, Mapping) else {"name": str(value)})
    return rows


def _catalog_payload(config: Mapping[str, Any]) -> dict[str, Any] | None:
    candidate = (
        config.get("feature_catalog") or config.get("feature_schema") or config.get("feature128")
    )
    if isinstance(candidate, Mapping):
        block = dict(candidate)
    elif "features" in config:
        schema_version = str(config.get("schema_version") or "feature128.v5")
        block = {
            "name": config.get("feature_set", schema_version),
            "version": config.get("feature_version", schema_version),
            "dimension": config.get("feature_dimension") or config.get("feature_count"),
            "features": config.get("features"),
        }
    else:
        return None
    if "features" not in block:
        block["features"] = block.get("definitions")
    return block


def seed_catalog(
    session: Session,
    config: Mapping[str, Any] | str | Path | None = None,
) -> object | None:
    """Idempotently seed an immutable feature catalog from framework JSON.

    Reusing a catalog ``name``/``version`` with different content is rejected.
    A catalog version therefore remains a trustworthy contract for stored JSON
    vectors on both SQLite and PostgreSQL.
    """

    from app.models import FeatureCatalog, FeatureDefinition

    framework_config = _load_json_config(config)
    block = _catalog_payload(framework_config)
    if block is None:
        return None

    rows = _flatten_feature_rows(block.get("features"))
    if not rows:
        raise ValueError("Feature catalog must contain at least one definition")
    dimension_value = block.get("dimension")
    dimension = int(dimension_value) if dimension_value is not None else len(rows)
    if dimension != len(rows):
        raise ValueError(f"Catalog dimension {dimension} does not match {len(rows)} definitions")

    name = str(block.get("name") or block.get("id") or "feature128.v5")
    version = str(block.get("version") or block.get("schema_version") or "1")
    canonical = json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    existing = session.scalar(
        select(FeatureCatalog).where(
            FeatureCatalog.name == name,
            FeatureCatalog.version == version,
        )
    )
    if existing is not None:
        if existing.checksum != checksum:
            raise ValueError(f"Catalog {name}@{version} already exists with a different checksum")
        return existing

    catalog = FeatureCatalog(
        name=name,
        version=version,
        dimension=dimension,
        checksum=checksum,
        is_active=bool(block.get("active", True)),
        metadata_json={
            "sequence": framework_config.get("sequence", {}),
            "source": block.get("source"),
        },
    )
    session.add(catalog)
    session.flush()

    seen_codes: set[str] = set()
    seen_ordinals: set[int] = set()
    for position, row in enumerate(rows, start=1):
        code = str(row.get("id") or row.get("code") or f"F{position:03d}")
        ordinal = int(row.get("ordinal") or position)
        if code in seen_codes or ordinal in seen_ordinals:
            raise ValueError(f"Duplicate feature code/ordinal: {code}/{ordinal}")
        seen_codes.add(code)
        seen_ordinals.add(ordinal)
        definition = FeatureDefinition(
            catalog_id=catalog.id,
            code=code,
            ordinal=ordinal,
            name=str(row.get("name") or code),
            group_name=str(row.get("group") or row.get("category") or "ungrouped"),
            value_kind=str(row.get("value_kind") or row.get("kind") or "numeric"),
            source=str(row.get("source") or row.get("source_constraint") or ""),
            description=str(
                row.get("description") or row.get("rule") or row.get("source_constraint") or ""
            ),
            constraints_json=dict(row.get("constraints") or {}),
        )
        session.add(definition)
    session.flush()
    return catalog


def init_db(
    bind: Engine | None = None,
    *,
    config: Mapping[str, Any] | str | Path | None = None,
    seed: bool = True,
) -> None:
    """Create all persistence tables and optionally seed the feature catalog."""

    from app import models as _models  # noqa: F401

    target = bind or engine
    Base.metadata.create_all(target)
    if seed:
        with session_scope(target) as session:
            if config is None:
                seed_source = settings.load_feature_catalog(required=True)
                seed_source["sequence"] = settings.load_sequence_config(required=True)
            else:
                seed_source = config
            seed_catalog(session, seed_source)
