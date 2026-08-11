"""PostgreSQL migration-chain coverage for legacy safe-update evidence."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

import app.settings as settings_module
from app.database import make_engine

BACKEND_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.postgres


def _alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


@pytest.fixture
def postgres_migration_database(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[str, str]]:
    database_url = os.getenv("TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_URL is not configured")

    admin_engine = make_engine(database_url)
    if admin_engine.dialect.name != "postgresql":
        admin_engine.dispose()
        pytest.fail("TEST_POSTGRES_URL must use PostgreSQL")

    schema_name = f"test_safe_update_migration_{uuid.uuid4().hex}"
    quoted_schema = admin_engine.dialect.identifier_preparer.quote_identifier(
        schema_name
    )
    schema_created = False
    try:
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE SCHEMA {quoted_schema}")
        schema_created = True

        scoped_url = make_url(database_url).update_query_dict(
            {"options": f"-csearch_path={schema_name}"}
        )
        scoped_database_url = scoped_url.render_as_string(hide_password=False)
        monkeypatch.setattr(
            settings_module,
            "settings",
            replace(settings_module.settings, database_url=scoped_database_url),
        )
        yield scoped_database_url, schema_name
    finally:
        if schema_created:
            with admin_engine.begin() as connection:
                connection.exec_driver_sql(f"DROP SCHEMA {quoted_schema} CASCADE")
        admin_engine.dispose()


def _seed_legacy_candidate(
    engine: sa.Engine,
) -> tuple[uuid.UUID, uuid.UUID, date]:
    metadata = sa.MetaData()
    metadata.reflect(engine)
    now = datetime.now(UTC)
    organization_id = uuid.uuid4()
    user_id = uuid.uuid4()
    role_id = uuid.uuid4()
    assignment_id = uuid.uuid4()
    profile_id = uuid.uuid4()
    assessment_id = uuid.uuid4()
    candidate_id = uuid.uuid4()
    candidate_day = date(2026, 1, 1)
    quarantine_until = candidate_day + timedelta(days=7)

    with engine.begin() as connection:
        connection.execute(
            metadata.tables["organizations"].insert(),
            {
                "id": organization_id,
                "slug": f"legacy-pg-{organization_id.hex}",
                "name": "Legacy PostgreSQL Org",
                "timezone": "UTC",
                "retention_policy": {},
                "created_at": now,
                "updated_at": now,
            },
        )
        connection.execute(
            metadata.tables["users"].insert(),
            {
                "id": user_id,
                "organization_id": organization_id,
                "external_user_id": f"U-PG-{user_id.hex}",
                "display_name": "Legacy PostgreSQL User",
                "status": "active",
                "first_seen": None,
                "last_seen": None,
                "attributes_json": {},
                "created_at": now,
                "updated_at": now,
            },
        )
        connection.execute(
            metadata.tables["roles"].insert(),
            {
                "id": role_id,
                "organization_id": organization_id,
                "code": f"PG-{role_id.hex}",
                "name": "Legacy PostgreSQL Role",
                "family": None,
                "is_unknown": False,
                "attributes_json": {},
                "created_at": now,
                "updated_at": now,
            },
        )
        connection.execute(
            metadata.tables["role_assignments"].insert(),
            {
                "id": assignment_id,
                "user_id": user_id,
                "role_id": role_id,
                "valid_from": date(2025, 1, 1),
                "valid_to": None,
                "source_snapshot_date": None,
                "source_snapshot": None,
                "source_checksum": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        connection.execute(
            metadata.tables["reference_profiles"].insert(),
            {
                "id": profile_id,
                "organization_id": organization_id,
                "branch": "feature",
                "level": "person",
                "scope_key": f"person:{assignment_id}",
                "user_id": user_id,
                "role_id": None,
                "role_assignment_id": assignment_id,
                "model_version": "legacy-model",
                "config_version": "framework.v4",
                "catalog_version": "feature128.v5",
                "fitted_from": date(2025, 1, 1),
                "fitted_through": date(2025, 12, 31),
                "support_days": 30,
                "support_users": 1,
                "support_transitions": 0,
                "coverage": 1.0,
                "support_json": {},
                "statistics_json": {},
                "calibrator_json": {
                    "method": "empirical_cdf.v1",
                    "sorted_scores": [0.1],
                },
                "is_frozen": False,
                "checksum": "a" * 64,
                "artifact_id": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        connection.execute(
            metadata.tables["risk_assessments"].insert(),
            {
                "id": assessment_id,
                "organization_id": organization_id,
                "user_id": user_id,
                "day": candidate_day,
                "role_assignment_id": assignment_id,
                "split": "production",
                "status": "no_score",
                "feature_score_id": None,
                "sequence_score_id": None,
                "feature_weight": 0.0,
                "sequence_weight": 0.0,
                "risk": None,
                "threshold": None,
                "is_alert": False,
                "model_version": "legacy-model",
                "config_version": "framework.v4",
                "scoring_run_id": "legacy-pg-run",
                "fusion_evidence": {},
                "created_at": now,
            },
        )
        connection.execute(
            metadata.tables["safe_update_candidates"].insert(),
            {
                "id": candidate_id,
                "organization_id": organization_id,
                "user_id": user_id,
                "role_assignment_id": assignment_id,
                "reference_profile_id": profile_id,
                "source_assessment_id": assessment_id,
                "branch": "feature",
                "candidate_day": candidate_day,
                "quarantine_until": quarantine_until,
                "status": "candidate",
                "reason_codes": [],
                "influence_cap": 0.05,
                "applied_at": None,
                "before_checksum": None,
                "after_checksum": None,
                "created_at": now,
                "updated_at": now,
            },
        )

    return profile_id, candidate_id, quarantine_until


def test_alembic_0002_to_head_corrects_legacy_state_on_postgresql(
    postgres_migration_database: tuple[str, str],
) -> None:
    database_url, schema_name = postgres_migration_database
    config = _alembic_config(database_url)
    command.upgrade(config, "0002")

    engine = sa.create_engine(database_url, poolclass=sa.pool.NullPool)
    try:
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT current_schema()")) == schema_name
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == "0002"
        profile_id, candidate_id, quarantine_until = _seed_legacy_candidate(engine)
    finally:
        engine.dispose()

    command.upgrade(config, "head")

    engine = sa.create_engine(database_url, poolclass=sa.pool.NullPool)
    try:
        metadata = sa.MetaData()
        metadata.reflect(engine)
        profiles = metadata.tables["reference_profiles"]
        candidates = metadata.tables["safe_update_candidates"]
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == "0004"
            profile = connection.execute(
                sa.select(profiles).where(profiles.c.id == profile_id)
            ).mappings().one()
            candidate = connection.execute(
                sa.select(candidates).where(candidates.c.id == candidate_id)
            ).mappings().one()
            constraints = {
                item["name"]: str(item.get("sqltext", "")).lower()
                for item in sa.inspect(connection).get_check_constraints(
                    "reference_profiles"
                )
            }
            release_constraints = {
                item["name"]: str(item.get("sqltext", "")).lower()
                for item in sa.inspect(connection).get_check_constraints(
                    "reference_releases"
                )
            }

        assert profile["policy_version"] == "framework.v4"
        assert profile["release_kind"] == "legacy"
        assert candidate["policy_version"] == "framework.v4.safe_update.v1"
        assert candidate["eligible_on"] == quarantine_until + timedelta(days=1)
        release_kind_constraint = constraints[
            "ck_reference_profiles_reference_profile_release_kind"
        ]
        assert all(
            value in release_kind_constraint
            for value in ("legacy", "bootstrap", "incremental")
        )
        materialized_kind_constraint = release_constraints[
            "ck_reference_releases_reference_release_materialized_kind"
        ]
        assert "legacy" not in materialized_kind_constraint
        assert all(
            value in materialized_kind_constraint
            for value in ("bootstrap", "incremental")
        )

        with pytest.raises(sa.exc.IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    profiles.update()
                    .where(profiles.c.id == profile_id)
                    .values(release_kind="unsupported")
                )
    finally:
        engine.dispose()
