"""Migration regression tests for legacy safe-update rows."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config

import app.settings as settings_module

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _upgrade_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_0003_preserves_legacy_candidate_without_fabricating_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "legacy-safe-update.db"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    monkeypatch.setattr(
        settings_module,
        "settings",
        replace(settings_module.settings, database_url=database_url),
    )
    config = _upgrade_config(database_url)
    command.upgrade(config, "0002")

    engine = sa.create_engine(database_url)
    metadata = sa.MetaData()
    metadata.reflect(engine)
    now = datetime.now(UTC)
    organization_id = uuid.uuid4().hex
    user_id = uuid.uuid4().hex
    role_id = uuid.uuid4().hex
    assignment_id = uuid.uuid4().hex
    profile_id = uuid.uuid4().hex
    assessment_id = uuid.uuid4().hex
    candidate_id = uuid.uuid4().hex
    candidate_day = date(2026, 1, 1)
    quarantine_until = candidate_day + timedelta(days=7)

    with engine.begin() as connection:
        connection.execute(
            metadata.tables["organizations"].insert(),
            {
                "id": organization_id,
                "slug": "legacy-org",
                "name": "Legacy Org",
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
                "external_user_id": "U-LEGACY",
                "display_name": "Legacy User",
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
                "code": "LEGACY",
                "name": "Legacy Role",
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
                "scoring_run_id": "legacy-run",
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
    engine.dispose()

    command.upgrade(config, "0003")

    engine = sa.create_engine(database_url)
    migrated = sa.MetaData()
    migrated.reflect(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
        candidate = connection.execute(
            sa.select(migrated.tables["safe_update_candidates"]).where(
                migrated.tables["safe_update_candidates"].c.id == candidate_id
            )
        ).mappings().one()
        profile = connection.execute(
            sa.select(migrated.tables["reference_profiles"]).where(
                migrated.tables["reference_profiles"].c.id == profile_id
            )
        ).mappings().one()

        assert candidate["policy_version"] == "framework.v4.safe_update.v1"
        assert candidate["model_version"] == "legacy-model"
        assert candidate["config_version"] == "framework.v4"
        assert candidate["eligible_on"] == quarantine_until
        assert candidate["influence_cap"] == 0.05
        assert candidate["reference_profile_id"] == profile_id
        assert candidate["source_branch_score_id"] is None
        assert candidate["admission_reference_profile_id"] is None
        assert candidate["admission_percentile"] is None
        assert candidate["support_contribution_json"] == {}

        assert profile["policy_version"] == "framework.v4"
        assert profile["parent_reference_profile_id"] is None
        assert profile["calibration_parent_profile_id"] is None
        assert profile["reference_version"] is None
        assert profile["release_kind"] is None
    engine.dispose()

    command.downgrade(config, "0002")
    engine = sa.create_engine(database_url)
    inspector = sa.inspect(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
    assert "policy_version" not in {
        column["name"]
        for column in inspector.get_columns("safe_update_candidates")
    }
    engine.dispose()
