from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.catalog import FEATURE_SCHEMA_VERSION, load_framework_config
from app.database import Base, init_db, make_engine
from app.domain.safe_update import SafeUpdatePolicy, empirical_percentile
from app.models import (
    AssessmentStatus,
    Branch,
    BranchScore,
    DataSplit,
    FeatureCatalog,
    PersonalReferenceAccumulator,
    ReferenceLevel,
    ReferenceProfile,
    ReferenceRelease,
    ReferenceReleaseKind,
    RiskAssessment,
    Role,
    RoleAssignment,
    SafeUpdateCandidate,
    ScoreStatus,
    ScoringWatermark,
    UpdateStatus,
    User,
    UserDayFeature,
    utc_now,
)
from app.schemas import BranchAssessmentInput
from app.services import (
    _record_safe_update_admission,
    ensure_default_organization,
    process_safe_updates,
)
from app.validation import sha256_json

MODEL_VERSION = "safe-flow.v1"
CONFIG_VERSION = "framework.v5"
FIRST_PRODUCTION_DAY = date(2011, 5, 18)


@pytest.fixture
def db() -> Session:
    engine = make_engine("sqlite+pysqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )
    session = factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _identity(
    db: Session,
    *,
    external_user_id: str,
) -> tuple[object, User, RoleAssignment]:
    organization = ensure_default_organization(db)
    role = db.scalar(
        select(Role).where(
            Role.organization_id == organization.id,
            Role.code == "ENGINEER",
        )
    )
    if role is None:
        role = Role(
            organization_id=organization.id,
            code="ENGINEER",
            name="Engineer",
        )
        db.add(role)
        db.flush()
    user = User(
        organization_id=organization.id,
        external_user_id=external_user_id,
        display_name=external_user_id,
    )
    db.add(user)
    db.flush()
    assignment = RoleAssignment(
        user_id=user.id,
        role_id=role.id,
        valid_from=date(2010, 1, 2),
        source_snapshot_date=date(2010, 1, 1),
    )
    db.add(assignment)
    db.flush()
    return organization, user, assignment


def _role_parent(
    db: Session,
    organization: object,
    assignment: RoleAssignment,
) -> ReferenceProfile:
    profile = ReferenceProfile(
        organization_id=organization.id,
        branch=Branch.FEATURE,
        level=ReferenceLevel.ROLE,
        scope_key=f"role:{assignment.role_id}",
        role_id=assignment.role_id,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        catalog_version=FEATURE_SCHEMA_VERSION,
        fitted_from=date(2010, 1, 2),
        fitted_through=date(2011, 5, 17),
        support_days=300,
        support_users=20,
        coverage=1.0,
        support_json={
            "peer_users": 20,
            "peer_user_days": 300,
            "recent_user_days": 100,
            "coverage": 1.0,
            "min_feature_observations": 200,
        },
        statistics_json={},
        calibrator_json={
            "method": "empirical_cdf.v1",
            "sorted_scores": [float(value) for value in range(10)],
        },
        policy_version=CONFIG_VERSION,
        is_frozen=True,
        checksum="a" * 64,
    )
    db.add(profile)
    db.flush()
    return profile


def _source_day(
    db: Session,
    organization: object,
    user: User,
    assignment: RoleAssignment,
    parent: ReferenceProfile,
    *,
    day: date,
    raw_score: float,
) -> tuple[UserDayFeature, BranchScore, RiskAssessment]:
    catalog = db.scalar(
        select(FeatureCatalog).where(FeatureCatalog.version == FEATURE_SCHEMA_VERSION)
    )
    assert catalog is not None
    input_checksum = sha256_json(
        {"user_id": user.external_user_id, "day": day, "raw_score": raw_score}
    )
    feature = UserDayFeature(
        organization_id=organization.id,
        user_id=user.id,
        day=day,
        role_assignment_id=assignment.id,
        catalog_id=catalog.id,
        split=DataSplit.PRODUCTION,
        values=[0.0] * 128,
        present_mask=[True] * 128,
        feature_count=128,
        is_observed_day=True,
        is_active_day=True,
        context_json={},
        input_checksum=input_checksum,
    )
    db.add(feature)
    db.flush()
    calibrated = empirical_percentile(
        raw_score,
        parent.calibrator_json["sorted_scores"],
    )
    branch_score = BranchScore(
        organization_id=organization.id,
        user_id=user.id,
        day=day,
        role_assignment_id=assignment.id,
        branch=Branch.FEATURE,
        status=ScoreStatus.SCORED,
        selected_level=ReferenceLevel.ROLE,
        reference_profile_id=parent.id,
        raw_score=raw_score,
        calibrated_score=calibrated,
        support_snapshot={},
        fallback_reasons=[],
        evidence={},
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        scoring_run_id=f"run-{user.external_user_id}-{day.isoformat()}",
    )
    db.add(branch_score)
    db.flush()
    assessment = RiskAssessment(
        organization_id=organization.id,
        user_id=user.id,
        day=day,
        role_assignment_id=assignment.id,
        split=DataSplit.PRODUCTION,
        status=AssessmentStatus.SCORED,
        feature_score_id=branch_score.id,
        feature_weight=1.0,
        sequence_weight=0.0,
        risk=calibrated,
        threshold=0.95,
        is_alert=False,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        scoring_run_id=branch_score.scoring_run_id,
        fusion_evidence={},
    )
    db.add(assessment)
    db.flush()
    return feature, branch_score, assessment


def _accepted_candidate(
    db: Session,
    organization: object,
    user: User,
    assignment: RoleAssignment,
    accumulator: PersonalReferenceAccumulator,
    parent: ReferenceProfile,
    *,
    day: date,
) -> SafeUpdateCandidate:
    feature, score, assessment = _source_day(
        db,
        organization,
        user,
        assignment,
        parent,
        day=day,
        raw_score=7.0,
    )
    candidate = SafeUpdateCandidate(
        organization_id=organization.id,
        user_id=user.id,
        role_assignment_id=assignment.id,
        reference_profile_id=accumulator.active_reference_profile_id,
        source_assessment_id=assessment.id,
        accumulator_id=accumulator.id,
        source_branch_score_id=score.id,
        source_feature_id=feature.id,
        source_input_checksum=feature.input_checksum,
        admission_reference_profile_id=parent.id,
        admission_reference_checksum=parent.checksum,
        admission_percentile=0.8,
        admission_threshold=0.9,
        branch=Branch.FEATURE,
        candidate_day=day,
        quarantine_until=day + timedelta(days=30),
        eligible_on=day + timedelta(days=31),
        status=UpdateStatus.ACCEPTED,
        reason_codes=["QUARANTINE_AND_WATERMARKS_PASSED"],
        support_contribution_json={
            "active_day": True,
            "feature_dimension": 128,
            "present_ordinals": list(range(1, 129)),
        },
        policy_version="framework.v5.safe_update.v1",
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        influence_cap=0.02,
        decision_at=utc_now(),
    )
    db.add(candidate)
    db.flush()
    return candidate


def _watermarks(
    db: Session,
    organization: object,
    *,
    start_offset: int,
    end_offset: int,
) -> None:
    for offset in range(start_offset, end_offset + 1):
        day = FIRST_PRODUCTION_DAY + timedelta(days=offset)
        checksum = sha256_json({"day": day})
        db.add(
            ScoringWatermark(
                organization_id=organization.id,
                day=day,
                model_version=MODEL_VERSION,
                config_version=CONFIG_VERSION,
                expected_assessments=0,
                persisted_assessments=0,
                universe_checksum=checksum,
                assessment_set_checksum=checksum,
                completed_at=utc_now(),
            )
        )
    db.flush()


def test_safe_update_v5_end_to_end(db: Session) -> None:
    organization, user, assignment = _identity(db, external_user_id="U001")
    _, reject_user, reject_assignment = _identity(db, external_user_id="U002")
    parent = _role_parent(db, organization, assignment)
    framework = load_framework_config()
    policy = SafeUpdatePolicy.from_framework(framework)

    _, reject_score, reject_assessment = _source_day(
        db,
        organization,
        reject_user,
        reject_assignment,
        parent,
        day=FIRST_PRODUCTION_DAY,
        raw_score=8.0,
    )
    rejected_at_boundary = _record_safe_update_admission(
        db,
        organization,
        reject_user,
        reject_assignment,
        assessment=reject_assessment,
        branch_score=reject_score,
        branch_input=BranchAssessmentInput(raw_score=8.0),
        profiles={ReferenceLevel.ROLE: parent},
        role_known=True,
        framework_config=framework,
        policy=policy,
        actor="test-scorer",
        request_id="strict-boundary",
    )
    assert rejected_at_boundary is None

    _, accepted_score, accepted_assessment = _source_day(
        db,
        organization,
        user,
        assignment,
        parent,
        day=FIRST_PRODUCTION_DAY,
        raw_score=7.0,
    )
    first_candidate = _record_safe_update_admission(
        db,
        organization,
        user,
        assignment,
        assessment=accepted_assessment,
        branch_score=accepted_score,
        branch_input=BranchAssessmentInput(raw_score=7.0),
        profiles={ReferenceLevel.ROLE: parent},
        role_known=True,
        framework_config=framework,
        policy=policy,
        actor="test-scorer",
        request_id="cold-start",
    )
    assert first_candidate is not None
    assert first_candidate.admission_percentile == pytest.approx(0.8)
    accumulator = db.get(PersonalReferenceAccumulator, first_candidate.accumulator_id)
    assert accumulator is not None
    assert accumulator.active_reference_profile_id is None

    no_watermark = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        limit=1_000,
        actor="test-scorer",
        request_id="no-watermark",
    )
    assert no_watermark["pending"] == 1
    assert no_watermark["accepted"] == 0

    for index in range(1, 60):
        offset = (index * 89) // 59
        _accepted_candidate(
            db,
            organization,
            user,
            assignment,
            accumulator,
            parent,
            day=FIRST_PRODUCTION_DAY + timedelta(days=offset),
        )
    first_incremental = _accepted_candidate(
        db,
        organization,
        user,
        assignment,
        accumulator,
        parent,
        day=FIRST_PRODUCTION_DAY + timedelta(days=90),
    )
    deferred_then_tampered = _accepted_candidate(
        db,
        organization,
        user,
        assignment,
        accumulator,
        parent,
        day=FIRST_PRODUCTION_DAY + timedelta(days=91),
    )
    _watermarks(db, organization, start_offset=0, end_offset=119)
    bootstrap = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        limit=1_000,
        actor="test-scorer",
        request_id="bootstrap",
    )
    assert bootstrap["accepted"] == 1
    assert bootstrap["applied"] == 60
    assert bootstrap["deferred"] == 2

    db.refresh(accumulator)
    bootstrap_profile = db.get(
        ReferenceProfile,
        accumulator.active_reference_profile_id,
    )
    assert bootstrap_profile is not None
    assert bootstrap_profile.release_kind is ReferenceReleaseKind.BOOTSTRAP
    assert bootstrap_profile.reference_version == 1
    assert bootstrap_profile.calibrator_json["method"] == "parent_shrunk_ecdf.v1"
    assert len(bootstrap_profile.calibrator_json["personal_sorted_scores"]) == 60
    assert parent.checksum == "a" * 64

    _watermarks(db, organization, start_offset=120, end_offset=128)
    incremental = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        limit=1_000,
        actor="test-scorer",
        request_id="incremental",
    )
    assert incremental["applied"] == 1
    assert incremental["deferred"] == 1
    db.refresh(first_incremental)
    assert first_incremental.status is UpdateStatus.APPLIED

    db.refresh(accumulator)
    incremental_profile = db.get(
        ReferenceProfile,
        accumulator.active_reference_profile_id,
    )
    assert incremental_profile is not None
    assert incremental_profile.release_kind is ReferenceReleaseKind.INCREMENTAL
    assert incremental_profile.reference_version == 2
    release = db.scalar(
        select(ReferenceRelease).where(
            ReferenceRelease.child_reference_profile_id == incremental_profile.id
        )
    )
    assert release is not None
    assert release.applied_candidate_count == 1
    assert release.parent_support == 60
    assert release.influence_ratio == pytest.approx(1 / 60)

    tampered_input = db.get(
        UserDayFeature,
        deferred_then_tampered.source_feature_id,
    )
    assert tampered_input is not None
    tampered_input.input_checksum = "f" * 64
    db.flush()
    _watermarks(db, organization, start_offset=129, end_offset=135)
    toctou = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        limit=1_000,
        actor="test-scorer",
        request_id="toctou",
    )
    assert toctou["applied"] == 0
    assert toctou["rejected"] == 1
    assert toctou["deferred"] == 0
    db.refresh(deferred_then_tampered)
    assert deferred_then_tampered.status is UpdateStatus.REJECTED
    assert "SOURCE_USER_DAY_CHANGED" in deferred_then_tampered.reason_codes
