"""Persistence integration tests against the portable SQLite backend."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from app.database import Base, init_db, make_engine, seed_catalog
from app.models import (
    Alert,
    AlertStatus,
    AssessmentStatus,
    AuditLog,
    Branch,
    BranchScore,
    DataSplit,
    FeatureCatalog,
    FeatureDefinition,
    ImmutableRecordError,
    Organization,
    ReferenceLevel,
    ReferenceProfile,
    RiskAssessment,
    Role,
    RoleAssignment,
    ScoreStatus,
    User,
    UserDayFeature,
    UserDaySequence,
)
from app.services import role_epoch_anchor


@pytest.fixture
def db_engine():
    test_engine = make_engine("sqlite+pysqlite:///:memory:")
    init_db(test_engine, seed=False)
    try:
        yield test_engine
    finally:
        Base.metadata.drop_all(test_engine)
        test_engine.dispose()


def _identity_graph(session: Session) -> tuple[Organization, User, Role, RoleAssignment]:
    organization = Organization(slug="acme", name="ACME", timezone="UTC")
    user = User(
        organization=organization,
        external_user_id="U001",
        display_name="Test User",
    )
    role = Role(organization=organization, code="ENGINEER", name="Engineer")
    session.add_all([organization, user, role])
    session.flush()
    assignment = RoleAssignment(
        user_id=user.id,
        role_id=role.id,
        valid_from=date(2010, 1, 1),
        valid_to=date(2010, 2, 1),
    )
    session.add(assignment)
    session.flush()
    return organization, user, role, assignment


def test_init_db_seeds_an_idempotent_feature_catalog(db_engine, tmp_path) -> None:
    config = {
        "feature_catalog": {
            "name": "feature2.test",
            "version": "1",
            "dimension": 2,
            "features": [
                {
                    "id": "L01",
                    "name": "logon_count",
                    "group": "logon",
                    "kind": "count",
                },
                {
                    "id": "H01",
                    "name": "http_request_count",
                    "group": "http",
                    "kind": "count",
                },
            ],
        },
        "sequence": {"vocabulary": ["LOGON", "HTTP"]},
    }
    config_path = tmp_path / "framework.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    init_db(db_engine, config=config_path)
    with Session(db_engine) as session:
        catalog = session.scalar(select(FeatureCatalog))
        assert catalog is not None
        assert catalog.dimension == 2
        assert [item.code for item in catalog.definitions] == ["L01", "H01"]
        assert catalog.metadata_json["sequence"]["vocabulary"] == [
            "LOGON",
            "HTTP",
        ]
        seed_catalog(session, config_path)
        session.commit()
        assert session.scalar(select(func.count(FeatureCatalog.id))) == 1
        assert session.scalar(select(func.count(FeatureDefinition.id))) == 2


def test_effective_role_assignments_are_half_open_and_cannot_overlap(db_engine) -> None:
    with Session(db_engine) as session:
        organization, user, _, first = _identity_graph(session)
        second_role = Role(
            organization_id=organization.id,
            code="MANAGER",
            name="Manager",
        )
        session.add(second_role)
        session.flush()
        adjacent = RoleAssignment(
            user_id=user.id,
            role_id=second_role.id,
            valid_from=first.valid_to,
            valid_to=None,
        )
        session.add(adjacent)
        session.commit()

        overlapping = RoleAssignment(
            user_id=user.id,
            role_id=second_role.id,
            valid_from=date(2010, 1, 15),
            valid_to=date(2010, 1, 20),
        )
        session.add(overlapping)
        with pytest.raises(ValueError, match="cannot overlap"):
            session.flush()


def test_role_epoch_continues_across_adjacent_same_role_ldap_rows(db_engine) -> None:
    with Session(db_engine) as session:
        organization, user, role, first = _identity_graph(session)
        second = RoleAssignment(
            user_id=user.id,
            role_id=role.id,
            valid_from=date(2010, 2, 1),
            valid_to=date(2010, 3, 1),
        )
        new_role = Role(
            organization=organization,
            code="MANAGER",
            name="Manager",
        )
        session.add_all([second, new_role])
        session.flush()
        changed = RoleAssignment(
            user_id=user.id,
            role_id=new_role.id,
            valid_from=date(2010, 3, 1),
            valid_to=None,
        )
        session.add(changed)
        session.flush()

        assert role_epoch_anchor(session, second).id == first.id
        assert role_epoch_anchor(session, changed).id == changed.id


def test_feature_and_sequence_shapes_are_validated(db_engine) -> None:
    with Session(db_engine) as session:
        organization, user, _, assignment = _identity_graph(session)
        catalog = FeatureCatalog(
            name="feature2.test",
            version="1",
            dimension=2,
            checksum="a" * 64,
        )
        session.add(catalog)
        session.flush()

        vector = UserDayFeature(
            organization_id=organization.id,
            user_id=user.id,
            day=date(2010, 1, 2),
            role_assignment_id=assignment.id,
            catalog_id=catalog.id,
            split=DataSplit.TRAIN,
            values=[1.0, 0.0],
            present_mask=[True, False],
            input_checksum="b" * 64,
        )
        sequence = UserDaySequence(
            organization_id=organization.id,
            user_id=user.id,
            day=date(2010, 1, 2),
            role_assignment_id=assignment.id,
            split=DataSplit.TRAIN,
            vocabulary_version="tokens.v1",
            tokens=["LOGON", "HTTP"],
            pc_contexts=["own", "own"],
            calendar_contexts=["WEEKDAY", "WEEKDAY"],
            gap_buckets=["0-1", "1-5"],
            event_uids=["event-1", "event-2"],
            seq_len=2,
            truncated=False,
            input_checksum="c" * 64,
        )
        session.add_all([vector, sequence])
        session.commit()
        assert vector.feature_count == 2
        assert sequence.stored_len == 2

        invalid = UserDayFeature(
            organization_id=organization.id,
            user_id=user.id,
            day=date(2010, 1, 3),
            catalog_id=catalog.id,
            split=DataSplit.TRAIN,
            values=[1.0],
            present_mask=[True],
            input_checksum="d" * 64,
        )
        session.add(invalid)
        with pytest.raises(ValueError, match="catalog requires 2"):
            session.flush()


def test_canonical_schema_has_no_label_columns_and_payload_rejects_answer_keys() -> None:
    from app.models import CanonicalEvent

    columns = set(inspect(CanonicalEvent).columns.keys())
    assert columns.isdisjoint(
        {"label", "labels", "scenario", "scenario_id", "insider_flag", "answer_key"}
    )
    with pytest.raises(ValueError, match="Ground-truth key"):
        CanonicalEvent(source_payload={"nested": {"scenario_id": "scenario-1"}})


def test_risk_and_audit_are_immutable_while_alert_is_mutable(db_engine) -> None:
    with Session(db_engine, expire_on_commit=False) as session:
        organization, user, _, assignment = _identity_graph(session)
        profile = ReferenceProfile(
            organization_id=organization.id,
            branch=Branch.FEATURE,
            level=ReferenceLevel.GLOBAL,
            scope_key="global",
            model_version="baseline.v1",
            config_version="config.v1",
            catalog_version="feature128.v5",
            fitted_through=date(2010, 5, 31),
            support_days=10_000,
            support_users=200,
            support_transitions=0,
            coverage=1.0,
            is_frozen=True,
            checksum="e" * 64,
        )
        session.add(profile)
        session.flush()
        score = BranchScore(
            organization_id=organization.id,
            user_id=user.id,
            day=date(2010, 6, 1),
            role_assignment_id=assignment.id,
            branch=Branch.FEATURE,
            status=ScoreStatus.SCORED,
            selected_level=ReferenceLevel.GLOBAL,
            reference_profile_id=profile.id,
            raw_score=2.0,
            calibrated_score=0.97,
            model_version="baseline.v1",
            config_version="config.v1",
            scoring_run_id="run-1",
        )
        session.add(score)
        session.flush()
        assessment = RiskAssessment(
            organization_id=organization.id,
            user_id=user.id,
            day=date(2010, 6, 1),
            role_assignment_id=assignment.id,
            split=DataSplit.VALIDATION,
            status=AssessmentStatus.SCORED,
            feature_score_id=score.id,
            feature_weight=1.0,
            sequence_weight=0.0,
            risk=0.97,
            threshold=0.95,
            is_alert=True,
            model_version="baseline.v1",
            config_version="config.v1",
            scoring_run_id="run-1",
        )
        session.add(assessment)
        session.flush()
        alert = Alert(
            organization_id=organization.id,
            assessment_id=assessment.id,
        )
        audit = AuditLog(
            organization_id=organization.id,
            actor="scorer",
            action="assessment.created",
            entity_type="risk_assessment",
            entity_id=str(assessment.id),
            event_hash="f" * 64,
        )
        session.add_all([alert, audit])
        session.commit()

        alert.status = AlertStatus.IN_REVIEW
        alert.assignee = "analyst@example.test"
        session.commit()
        assert alert.status is AlertStatus.IN_REVIEW

        assessment.risk = 0.5
        with pytest.raises(ImmutableRecordError, match="cannot be updated"):
            session.flush()
        session.rollback()

        audit.action = "tampered"
        with pytest.raises(ImmutableRecordError, match="cannot be updated"):
            session.flush()
        session.rollback()


def test_safe_update_quarantine_constraint_is_enforced(db_engine) -> None:
    """The portable database itself rejects quarantine dates before candidate day."""

    from app.models import SafeUpdateCandidate

    with Session(db_engine) as session:
        organization, user, _, assignment = _identity_graph(session)
        profile = ReferenceProfile(
            organization_id=organization.id,
            branch=Branch.FEATURE,
            level=ReferenceLevel.PERSON,
            scope_key=f"person:{assignment.id}",
            user_id=user.id,
            role_assignment_id=assignment.id,
            model_version="baseline.v1",
            config_version="config.v1",
            catalog_version="feature128.v5",
            fitted_through=date(2010, 1, 31),
            checksum="1" * 64,
        )
        session.add(profile)
        session.flush()
        assessment = RiskAssessment(
            organization_id=organization.id,
            user_id=user.id,
            day=date(2010, 2, 1),
            role_assignment_id=assignment.id,
            split=DataSplit.TRAIN,
            status=AssessmentStatus.NO_SCORE,
            feature_weight=0.0,
            sequence_weight=0.0,
            model_version="baseline.v1",
            config_version="config.v1",
            scoring_run_id="run-2",
        )
        session.add(assessment)
        session.flush()
        candidate = SafeUpdateCandidate(
            organization_id=organization.id,
            user_id=user.id,
            role_assignment_id=assignment.id,
            reference_profile_id=profile.id,
            source_assessment_id=assessment.id,
            branch=Branch.FEATURE,
            candidate_day=date(2010, 2, 1),
            quarantine_until=date(2010, 2, 1) - timedelta(days=1),
        )
        session.add(candidate)
        with pytest.raises(Exception, match="safe_update_quarantine_after_day"):
            session.flush()
