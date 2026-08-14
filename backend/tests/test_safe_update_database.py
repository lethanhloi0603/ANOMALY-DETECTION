"""Focused persistence tests for the safe-update v5 state foundation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import Base, init_db, make_engine
from app.models import (
    AssessmentStatus,
    Branch,
    DataSplit,
    ImmutableRecordError,
    Organization,
    PersonalAccumulatorStatus,
    PersonalReferenceAccumulator,
    ReferenceLevel,
    ReferenceProfile,
    RiskAssessment,
    Role,
    RoleAssignment,
    SafeUpdateCandidate,
    ScoringWatermark,
    User,
)


@pytest.fixture
def db_engine():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    init_db(engine, seed=False)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _identity_and_parent(
    session: Session,
) -> tuple[
    Organization,
    User,
    RoleAssignment,
    ReferenceProfile,
]:
    organization = Organization(slug="safe-v5", name="Safe V5", timezone="UTC")
    user = User(
        organization=organization,
        external_user_id="U-V5",
        display_name="Safe Update User",
    )
    role = Role(organization=organization, code="ENGINEER", name="Engineer")
    session.add_all([organization, user, role])
    session.flush()
    assignment = RoleAssignment(
        user_id=user.id,
        role_id=role.id,
        valid_from=date(2026, 1, 1),
    )
    session.add(assignment)
    session.flush()
    parent = ReferenceProfile(
        organization_id=organization.id,
        branch=Branch.FEATURE,
        level=ReferenceLevel.ROLE,
        scope_key=f"role:{role.id}",
        role_id=role.id,
        model_version="model.v5",
        config_version="framework.v5",
        catalog_version="feature128.v5",
        fitted_through=date(2025, 12, 31),
        support_days=500,
        support_users=25,
        support_transitions=0,
        coverage=1.0,
        calibrator_json={"method": "empirical_cdf.v1", "sorted_scores": [0.1, 0.2]},
        is_frozen=True,
        checksum="a" * 64,
    )
    session.add(parent)
    session.flush()
    return organization, user, assignment, parent


def test_accumulator_requires_a_frozen_non_person_admission_parent(db_engine) -> None:
    with Session(db_engine) as session:
        organization, user, assignment, parent = _identity_and_parent(session)
        accumulator = PersonalReferenceAccumulator(
            organization_id=organization.id,
            user_id=user.id,
            role_assignment_id=assignment.id,
            branch=Branch.FEATURE,
            model_version="model.v5",
            config_version="framework.v5",
            catalog_version="feature128.v5",
            admission_reference_profile_id=parent.id,
            status=PersonalAccumulatorStatus.WARMING,
        )
        session.add(accumulator)
        session.commit()

        assert accumulator.active_reference_profile_id is None
        assert accumulator.release_sequence == 0
        assert parent.policy_version == "framework.v5"


def test_watermark_is_complete_and_immutable(db_engine) -> None:
    with Session(db_engine) as session:
        organization, _, _, parent = _identity_and_parent(session)
        session.commit()
        organization_id = organization.id
        parent_id = parent.id
        incomplete = ScoringWatermark(
            organization_id=organization_id,
            day=date(2026, 2, 1),
            model_version="model.v5",
            config_version="framework.v5",
            expected_assessments=2,
            persisted_assessments=1,
            universe_checksum="b" * 64,
            assessment_set_checksum="c" * 64,
            completed_at=datetime.now(UTC),
        )
        session.add(incomplete)
        with pytest.raises(IntegrityError, match="scoring_watermark_complete"):
            session.flush()
        session.rollback()

        organization = session.get(Organization, organization_id)
        assert organization is not None
        watermark = ScoringWatermark(
            organization_id=organization.id,
            day=date(2026, 2, 1),
            model_version="model.v5",
            config_version="framework.v5",
            expected_assessments=2,
            persisted_assessments=2,
            universe_checksum="b" * 64,
            assessment_set_checksum="c" * 64,
            completed_at=datetime.now(UTC),
        )
        session.add(watermark)
        session.commit()

        watermark.assessment_set_checksum = "d" * 64
        with pytest.raises(ImmutableRecordError, match="cannot be updated"):
            session.flush()
        session.rollback()

        parent = session.get(ReferenceProfile, parent_id)
        assert parent is not None
        parent.checksum = "e" * 64
        with pytest.raises(ImmutableRecordError, match="cannot be updated"):
            session.flush()


def test_v5_candidate_cannot_exist_without_admission_evidence(db_engine) -> None:
    with Session(db_engine) as session:
        organization, user, assignment, _ = _identity_and_parent(session)
        assessment = RiskAssessment(
            organization_id=organization.id,
            user_id=user.id,
            day=date(2026, 2, 1),
            role_assignment_id=assignment.id,
            split=DataSplit.PRODUCTION,
            status=AssessmentStatus.NO_SCORE,
            feature_weight=0.0,
            sequence_weight=0.0,
            is_alert=False,
            model_version="model.v5",
            config_version="framework.v5",
            scoring_run_id="run-v5",
        )
        session.add(assessment)
        session.flush()
        candidate = SafeUpdateCandidate(
            organization_id=organization.id,
            user_id=user.id,
            role_assignment_id=assignment.id,
            source_assessment_id=assessment.id,
            branch=Branch.FEATURE,
            candidate_day=date(2026, 2, 1),
            quarantine_until=date(2026, 2, 1) + timedelta(days=30),
            eligible_on=date(2026, 2, 1) + timedelta(days=31),
            policy_version="framework.v5.safe_update.v1",
            model_version="model.v5",
            config_version="framework.v5",
        )
        session.add(candidate)
        with pytest.raises(ValueError, match="lacks evidence"):
            session.flush()
