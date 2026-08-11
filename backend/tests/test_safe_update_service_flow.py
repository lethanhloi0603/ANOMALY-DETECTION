from __future__ import annotations

import uuid
from copy import deepcopy
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
    Organization,
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
from app.validation import DomainValidationError, sha256_json

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
    organization: Organization | None = None,
) -> tuple[Organization, User, RoleAssignment]:
    if organization is None:
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
    is_alert: bool = False,
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
        is_alert=is_alert,
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
    omitted_offsets: set[int] | None = None,
) -> None:
    for offset in range(start_offset, end_offset + 1):
        if omitted_offsets is not None and offset in omitted_offsets:
            continue
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


def _framework_with_materialization(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
) -> tuple[dict[str, object], SafeUpdatePolicy]:
    framework = deepcopy(load_framework_config())
    framework["safe_personalized_update"]["release"][
        "materialization_enabled"
    ] = enabled
    monkeypatch.setattr("app.services.load_framework_config", lambda: framework)
    return framework, SafeUpdatePolicy.from_framework(framework)


def _admit_feature_candidate(
    db: Session,
    organization: object,
    user: User,
    assignment: RoleAssignment,
    parent: ReferenceProfile,
    *,
    day: date,
    framework: dict[str, object],
    policy: SafeUpdatePolicy,
    request_id: str,
) -> tuple[SafeUpdateCandidate, PersonalReferenceAccumulator]:
    _, score, assessment = _source_day(
        db,
        organization,
        user,
        assignment,
        parent,
        day=day,
        raw_score=7.0,
    )
    candidate = _record_safe_update_admission(
        db,
        organization,
        user,
        assignment,
        assessment=assessment,
        branch_score=score,
        branch_input=BranchAssessmentInput(raw_score=7.0),
        profiles={ReferenceLevel.ROLE: parent},
        role_known=True,
        framework_config=framework,
        policy=policy,
        actor="test-scorer",
        request_id=request_id,
    )
    assert candidate is not None
    accumulator = db.get(PersonalReferenceAccumulator, candidate.accumulator_id)
    assert accumulator is not None
    return candidate, accumulator


@pytest.mark.parametrize(
    ("config_materialization_enabled", "runtime_materialization_enabled"),
    [(False, True), (True, False)],
)
def test_safe_update_shadow_mode_keeps_accepted_candidate_deferred(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    config_materialization_enabled: bool,
    runtime_materialization_enabled: bool,
) -> None:
    organization, user, assignment = _identity(db, external_user_id="U-SHADOW")
    parent = _role_parent(db, organization, assignment)
    framework = deepcopy(load_framework_config())
    framework["safe_personalized_update"]["release"][
        "materialization_enabled"
    ] = config_materialization_enabled
    monkeypatch.setattr("app.services.load_framework_config", lambda: framework)
    policy = SafeUpdatePolicy.from_framework(framework)
    _, score, assessment = _source_day(
        db,
        organization,
        user,
        assignment,
        parent,
        day=FIRST_PRODUCTION_DAY,
        raw_score=7.0,
    )
    candidate = _record_safe_update_admission(
        db,
        organization,
        user,
        assignment,
        assessment=assessment,
        branch_score=score,
        branch_input=BranchAssessmentInput(raw_score=7.0),
        profiles={ReferenceLevel.ROLE: parent},
        role_known=True,
        framework_config=framework,
        policy=policy,
        actor="test-scorer",
        request_id="shadow-admission",
    )
    assert candidate is not None
    _watermarks(db, organization, start_offset=0, end_offset=30)

    result = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        runtime_materialization_enabled=runtime_materialization_enabled,
        limit=1_000,
        actor="test-scorer",
        request_id="shadow-process",
    )

    db.refresh(candidate)
    accumulator = db.get(PersonalReferenceAccumulator, candidate.accumulator_id)
    assert accumulator is not None
    assert result["accepted"] == 1
    assert result["applied"] == 0
    assert result["deferred"] == 1
    assert candidate.status is UpdateStatus.ACCEPTED
    assert accumulator.active_reference_profile_id is None
    assert db.scalar(select(ReferenceRelease).limit(1)) is None


@pytest.mark.parametrize(
    ("legacy_in_other_organization", "organization_legacy_pending"),
    [(False, 1), (True, 0)],
)
def test_global_legacy_gate_fails_before_mutation_when_legacy_is_pending(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    legacy_in_other_organization: bool,
    organization_legacy_pending: int,
) -> None:
    organization, current_user, current_assignment = _identity(
        db,
        external_user_id="U-CURRENT",
    )
    legacy_organization = organization
    if legacy_in_other_organization:
        legacy_organization = Organization(
            slug="legacy-other-org",
            name="Legacy Other Org",
            timezone="UTC",
            retention_policy={},
        )
        db.add(legacy_organization)
        db.flush()
    _, legacy_user, legacy_assignment = _identity(
        db,
        external_user_id="U-LEGACY",
        organization=legacy_organization,
    )
    parent = _role_parent(db, organization, current_assignment)
    legacy_parent = (
        _role_parent(db, legacy_organization, legacy_assignment)
        if legacy_in_other_organization
        else parent
    )
    framework = deepcopy(load_framework_config())
    framework["safe_personalized_update"]["release"][
        "materialization_enabled"
    ] = True
    monkeypatch.setattr("app.services.load_framework_config", lambda: framework)
    policy = SafeUpdatePolicy.from_framework(framework)

    _, current_score, current_assessment = _source_day(
        db,
        organization,
        current_user,
        current_assignment,
        parent,
        day=FIRST_PRODUCTION_DAY,
        raw_score=7.0,
    )
    current_candidate = _record_safe_update_admission(
        db,
        organization,
        current_user,
        current_assignment,
        assessment=current_assessment,
        branch_score=current_score,
        branch_input=BranchAssessmentInput(raw_score=7.0),
        profiles={ReferenceLevel.ROLE: parent},
        role_known=True,
        framework_config=framework,
        policy=policy,
        actor="test-scorer",
        request_id="current-admission",
    )
    assert current_candidate is not None

    _, _, legacy_assessment = _source_day(
        db,
        legacy_organization,
        legacy_user,
        legacy_assignment,
        legacy_parent,
        day=FIRST_PRODUCTION_DAY,
        raw_score=7.0,
    )
    legacy_candidate = SafeUpdateCandidate(
        organization_id=legacy_organization.id,
        user_id=legacy_user.id,
        role_assignment_id=legacy_assignment.id,
        reference_profile_id=legacy_parent.id,
        source_assessment_id=legacy_assessment.id,
        branch=Branch.FEATURE,
        candidate_day=FIRST_PRODUCTION_DAY,
        quarantine_until=FIRST_PRODUCTION_DAY + timedelta(days=7),
        eligible_on=FIRST_PRODUCTION_DAY + timedelta(days=7),
        status=UpdateStatus.CANDIDATE,
        reason_codes=[],
        policy_version="framework.v4.safe_update.v1",
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        influence_cap=0.05,
    )
    db.add(legacy_candidate)
    db.flush()
    _watermarks(db, organization, start_offset=0, end_offset=30)

    with pytest.raises(DomainValidationError) as exc_info:
        process_safe_updates(
            db,
            organization,
            model_version=MODEL_VERSION,
            config_version=CONFIG_VERSION,
            runtime_materialization_enabled=True,
            limit=1_000,
            actor="test-scorer",
            request_id="legacy-block",
        )

    assert exc_info.value.code == "SAFE_UPDATE_LEGACY_PENDING"
    assert exc_info.value.details == {
        "global_legacy_pending": True,
        "organization_legacy_pending": organization_legacy_pending,
    }
    db.refresh(current_candidate)
    db.refresh(legacy_candidate)
    assert current_candidate.status is UpdateStatus.CANDIDATE
    assert current_candidate.decision_at is None
    assert legacy_candidate.status is UpdateStatus.CANDIDATE
    assert db.scalar(select(ReferenceRelease).limit(1)) is None


def test_legacy_profile_kind_cannot_be_used_for_a_release_event(db: Session) -> None:
    release = ReferenceRelease(
        organization_id=uuid.uuid4(),
        accumulator_id=uuid.uuid4(),
        release_sequence=1,
        kind=ReferenceReleaseKind.LEGACY,
        parent_reference_profile_id=None,
        calibration_parent_profile_id=uuid.uuid4(),
        child_reference_profile_id=uuid.uuid4(),
        release_day=FIRST_PRODUCTION_DAY,
        parent_support=0,
        applied_candidate_count=1,
        influence_ratio=1.0,
        rolling_anchor_support=0,
        rolling_applied_count_before=0,
        policy_version="framework.v4",
        candidate_manifest_json={"candidate_ids": [str(uuid.uuid4())]},
        manifest_checksum="f" * 64,
    )
    db.add(release)

    with pytest.raises(
        ValueError,
        match="Legacy references cannot be materialized release events",
    ):
        db.flush()


def test_legacy_profile_kind_is_restricted_to_framework_v4(db: Session) -> None:
    organization = ensure_default_organization(db)
    profile = ReferenceProfile(
        organization_id=organization.id,
        branch=Branch.FEATURE,
        level=ReferenceLevel.GLOBAL,
        scope_key="global",
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        catalog_version=FEATURE_SCHEMA_VERSION,
        fitted_through=FIRST_PRODUCTION_DAY - timedelta(days=1),
        support_days=0,
        support_users=0,
        support_transitions=0,
        support_json={},
        statistics_json={},
        calibrator_json={},
        release_kind=ReferenceReleaseKind.LEGACY,
        policy_version=CONFIG_VERSION,
        is_frozen=True,
        checksum="e" * 64,
    )
    db.add(profile)

    with pytest.raises(
        ValueError,
        match="Legacy reference lineage is restricted to framework.v4 profiles",
    ):
        db.flush()


def test_safe_update_v5_end_to_end(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization, user, assignment = _identity(db, external_user_id="U001")
    _, reject_user, reject_assignment = _identity(db, external_user_id="U002")
    parent = _role_parent(db, organization, assignment)
    framework = deepcopy(load_framework_config())
    framework["safe_personalized_update"]["release"][
        "materialization_enabled"
    ] = True
    monkeypatch.setattr("app.services.load_framework_config", lambda: framework)
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
        runtime_materialization_enabled=True,
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
        runtime_materialization_enabled=True,
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
        runtime_materialization_enabled=True,
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
        runtime_materialization_enabled=True,
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


def test_alert_on_d_plus_30_rejects_candidate(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization, user, assignment = _identity(db, external_user_id="U-ALERT-D30")
    parent = _role_parent(db, organization, assignment)
    framework, policy = _framework_with_materialization(monkeypatch, enabled=False)
    candidate, _ = _admit_feature_candidate(
        db,
        organization,
        user,
        assignment,
        parent,
        day=FIRST_PRODUCTION_DAY,
        framework=framework,
        policy=policy,
        request_id="alert-d30-admission",
    )
    alert_day = FIRST_PRODUCTION_DAY + timedelta(days=30)
    _, _, alert_assessment = _source_day(
        db,
        organization,
        user,
        assignment,
        parent,
        day=alert_day,
        raw_score=9.0,
        is_alert=True,
    )
    assert alert_assessment.is_alert is True
    _watermarks(db, organization, start_offset=0, end_offset=30)

    result = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        runtime_materialization_enabled=True,
        limit=1_000,
        actor="test-scorer",
        request_id="alert-d30-process",
    )

    db.refresh(candidate)
    assert candidate.quarantine_until == alert_day
    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert result["applied"] == 0
    assert candidate.status is UpdateStatus.REJECTED
    assert "ALERT_IN_QUARANTINE_WINDOW" in candidate.reason_codes


def test_missing_one_quarantine_watermark_keeps_candidate_pending(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization, user, assignment = _identity(db, external_user_id="U-WATERMARK-GAP")
    parent = _role_parent(db, organization, assignment)
    framework, policy = _framework_with_materialization(monkeypatch, enabled=False)
    candidate, _ = _admit_feature_candidate(
        db,
        organization,
        user,
        assignment,
        parent,
        day=FIRST_PRODUCTION_DAY,
        framework=framework,
        policy=policy,
        request_id="watermark-gap-admission",
    )
    _watermarks(
        db,
        organization,
        start_offset=0,
        end_offset=30,
        omitted_offsets={15},
    )

    result = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        runtime_materialization_enabled=True,
        limit=1_000,
        actor="test-scorer",
        request_id="watermark-gap-process",
    )

    db.refresh(candidate)
    watermark_days = set(
        db.scalars(
            select(ScoringWatermark.day).where(
                ScoringWatermark.organization_id == organization.id
            )
        )
    )
    assert FIRST_PRODUCTION_DAY + timedelta(days=30) in watermark_days
    assert FIRST_PRODUCTION_DAY + timedelta(days=15) not in watermark_days
    assert len(watermark_days) == 30
    assert result["pending"] == 1
    assert result["accepted"] == 0
    assert result["rejected"] == 0
    assert candidate.status is UpdateStatus.CANDIDATE
    assert candidate.decision_at is None


def test_process_retry_does_not_create_second_reference_release(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization, user, assignment = _identity(db, external_user_id="U-RETRY")
    parent = _role_parent(db, organization, assignment)
    framework, policy = _framework_with_materialization(monkeypatch, enabled=True)
    _, accumulator = _admit_feature_candidate(
        db,
        organization,
        user,
        assignment,
        parent,
        day=FIRST_PRODUCTION_DAY,
        framework=framework,
        policy=policy,
        request_id="retry-admission",
    )
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
    _watermarks(db, organization, start_offset=0, end_offset=119)

    first = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        runtime_materialization_enabled=True,
        limit=1_000,
        actor="test-scorer",
        request_id="retry-first-process",
    )
    db.refresh(accumulator)
    first_profile_id = accumulator.active_reference_profile_id
    first_release_ids = list(
        db.scalars(
            select(ReferenceRelease.id).where(
                ReferenceRelease.accumulator_id == accumulator.id
            )
        )
    )

    retry = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        runtime_materialization_enabled=True,
        limit=1_000,
        actor="test-scorer",
        request_id="retry-second-process",
    )

    db.refresh(accumulator)
    retry_release_ids = list(
        db.scalars(
            select(ReferenceRelease.id).where(
                ReferenceRelease.accumulator_id == accumulator.id
            )
        )
    )
    assert first["applied"] == 60
    assert first_release_ids and len(first_release_ids) == 1
    assert retry["accepted"] == 0
    assert retry["rejected"] == 0
    assert retry["applied"] == 0
    assert retry_release_ids == first_release_ids
    assert accumulator.release_sequence == 1
    assert accumulator.active_reference_profile_id == first_profile_id


def test_role_epoch_change_rejects_and_new_epoch_uses_new_accumulator(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization, user, old_assignment = _identity(
        db,
        external_user_id="U-ROLE-EPOCH",
    )
    old_parent = _role_parent(db, organization, old_assignment)
    framework, policy = _framework_with_materialization(monkeypatch, enabled=False)
    old_candidate, old_accumulator = _admit_feature_candidate(
        db,
        organization,
        user,
        old_assignment,
        old_parent,
        day=FIRST_PRODUCTION_DAY,
        framework=framework,
        policy=policy,
        request_id="old-epoch-admission",
    )

    change_day = FIRST_PRODUCTION_DAY + timedelta(days=15)
    new_role = Role(
        organization_id=organization.id,
        code="MANAGER",
        name="Manager",
    )
    db.add(new_role)
    db.flush()
    old_assignment.valid_to = change_day
    db.flush()
    new_assignment = RoleAssignment(
        user_id=user.id,
        role_id=new_role.id,
        valid_from=change_day,
        source_snapshot_date=change_day,
    )
    db.add(new_assignment)
    db.flush()
    new_parent = _role_parent(db, organization, new_assignment)
    _watermarks(db, organization, start_offset=0, end_offset=30)

    result = process_safe_updates(
        db,
        organization,
        model_version=MODEL_VERSION,
        config_version=CONFIG_VERSION,
        runtime_materialization_enabled=True,
        limit=1_000,
        actor="test-scorer",
        request_id="role-epoch-process",
    )

    db.refresh(old_candidate)
    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert old_candidate.status is UpdateStatus.REJECTED
    assert "ROLE_EPOCH_CHANGED" in old_candidate.reason_codes

    new_candidate, new_accumulator = _admit_feature_candidate(
        db,
        organization,
        user,
        new_assignment,
        new_parent,
        day=FIRST_PRODUCTION_DAY + timedelta(days=31),
        framework=framework,
        policy=policy,
        request_id="new-epoch-admission",
    )
    assert new_candidate.role_assignment_id == new_assignment.id
    assert new_accumulator.role_assignment_id == new_assignment.id
    assert new_accumulator.id != old_accumulator.id
