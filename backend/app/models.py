"""Portable SQLAlchemy models for the operational insider-threat database.

Ground-truth labels intentionally do not exist in this schema. Evaluation data
belongs to the physically separate database described in ``data/evaluation/``.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import UTC, date, datetime
from enum import Enum as PyEnum
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
    inspect,
    or_,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship, validates

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


def json_type() -> JSON:
    return JSON().with_variant(JSONB(), "postgresql")


def enum_type(enum_class: type[PyEnum], name: str, length: int = 32) -> Enum:
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
        length=length,
    )


class UserStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class EventSource(StrEnum):
    LOGON = "logon"
    DEVICE = "device"
    FILE = "file"
    HTTP = "http"
    EMAIL = "email"


class IngestSource(StrEnum):
    LOGON = "logon"
    DEVICE = "device"
    FILE = "file"
    HTTP = "http"
    EMAIL = "email"
    LDAP = "ldap"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(StrEnum):
    CANONICAL_EVENTS = "canonical_events"
    FEATURES = "features"
    SEQUENCES = "sequences"
    ROLE_CONTEXT = "role_context"
    BRANCH_SCORES = "branch_scores"
    REFERENCE_PROFILE = "reference_profile"
    RISK_ASSESSMENTS = "risk_assessments"
    MODEL = "model"
    MANIFEST = "manifest"


class ArtifactStatus(StrEnum):
    STAGED = "staged"
    READY = "ready"
    INVALID = "invalid"
    ARCHIVED = "archived"


class PCContext(StrEnum):
    OWN = "own"
    SHARED = "shared"
    FOREIGN = "foreign"
    UNKNOWN = "unknown"


class LegacyTimeContext(StrEnum):
    """Initial-schema values retained only for persisted-row compatibility."""

    WORK = "work"
    AFTER = "after"
    WEEKEND = "weekend"


class DataSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


class Branch(StrEnum):
    FEATURE = "feature"
    SEQUENCE = "sequence"


class ReferenceLevel(StrEnum):
    PERSON = "person"
    ROLE = "role"
    GLOBAL = "global"


class ScoreStatus(StrEnum):
    SCORED = "scored"
    NO_SCORE = "no_score"


class AssessmentStatus(StrEnum):
    SCORED = "scored"
    NO_SCORE = "no_score"


class AlertStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_REVIEW = "in_review"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    DISMISSED = "dismissed"
    FALSE_POSITIVE = "false_positive"


class AlertSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UpdateStatus(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    APPLIED = "applied"


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "organizations"
    __table_args__ = (
        UniqueConstraint("slug", name="organization_slug"),
        CheckConstraint("length(slug) > 0", name="organization_slug_nonempty"),
    )

    slug: Mapped[str] = mapped_column(String(80), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    retention_policy: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )

    users: Mapped[list[User]] = relationship(back_populates="organization")
    roles: Mapped[list[Role]] = relationship(back_populates="organization")


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "external_user_id",
            name="user_external_id_per_organization",
        ),
        Index("ix_users_org_status", "organization_id", "status"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    external_user_id: Mapped[str] = mapped_column(String(160), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(240))
    status: Mapped[UserStatus] = mapped_column(
        enum_type(UserStatus, "user_status"), nullable=False, default=UserStatus.ACTIVE
    )
    first_seen: Mapped[date | None] = mapped_column(Date)
    last_seen: Mapped[date | None] = mapped_column(Date)
    attributes_json: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )

    organization: Mapped[Organization] = relationship(back_populates="users")
    role_assignments: Mapped[list[RoleAssignment]] = relationship(
        back_populates="user", order_by="RoleAssignment.valid_from"
    )


class Role(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "roles"
    __table_args__ = (
        UniqueConstraint("organization_id", "code", name="role_code_per_organization"),
        Index("ix_roles_org_family", "organization_id", "family"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    code: Mapped[str] = mapped_column(String(160), nullable=False)
    name: Mapped[str] = mapped_column(String(240), nullable=False)
    family: Mapped[str | None] = mapped_column(String(160))
    is_unknown: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attributes_json: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )

    organization: Mapped[Organization] = relationship(back_populates="roles")
    assignments: Mapped[list[RoleAssignment]] = relationship(back_populates="role")


class RoleAssignment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Effective-dated half-open role epoch: ``[valid_from, valid_to)``."""

    __tablename__ = "role_assignments"
    __table_args__ = (
        UniqueConstraint("user_id", "valid_from", name="role_assignment_start_per_user"),
        CheckConstraint(
            "valid_to IS NULL OR valid_to > valid_from",
            name="role_assignment_valid_range",
        ),
        CheckConstraint(
            "source_snapshot_date IS NULL OR source_snapshot_date <= valid_from",
            name="role_snapshot_not_future",
        ),
        Index(
            "ix_role_assignments_user_range",
            "user_id",
            "valid_from",
            "valid_to",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roles.id", ondelete="RESTRICT"), nullable=False
    )
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date)
    source_snapshot_date: Mapped[date | None] = mapped_column(Date)
    source_snapshot: Mapped[str | None] = mapped_column(String(300))
    source_checksum: Mapped[str | None] = mapped_column(String(128))

    user: Mapped[User] = relationship(back_populates="role_assignments")
    role: Mapped[Role] = relationship(back_populates="assignments")


class IngestionJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "idempotency_key",
            name="ingestion_idempotency_per_organization",
        ),
        UniqueConstraint(
            "organization_id",
            "source",
            "sha256",
            name="ingestion_content_per_source",
        ),
        CheckConstraint(
            "total_rows >= 0 AND processed_rows >= 0 AND rejected_rows >= 0",
            name="ingestion_nonnegative_counts",
        ),
        Index(
            "ix_ingestion_jobs_org_status_created",
            "organization_id",
            "status",
            "created_at",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    source: Mapped[IngestSource] = mapped_column(
        enum_type(IngestSource, "ingest_source"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        enum_type(JobStatus, "job_status"), nullable=False, default=JobStatus.PENDING
    )
    input_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(300))
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(160), nullable=False)
    schema_version: Mapped[str | None] = mapped_column(String(80))
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    min_event_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    max_event_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    progress_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    checkpoints: Mapped[list[IngestionCheckpoint]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[Artifact]] = relationship(back_populates="ingestion_job")


class IngestionCheckpoint(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_checkpoints"
    __table_args__ = (
        UniqueConstraint("job_id", "partition_key", name="checkpoint_partition_per_job"),
        CheckConstraint("row_offset >= 0", name="checkpoint_nonnegative_offset"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False
    )
    partition_key: Mapped[str] = mapped_column(String(240), nullable=False)
    row_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cursor_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)
    input_checksum: Mapped[str | None] = mapped_column(String(128))
    output_checksum: Mapped[str | None] = mapped_column(String(128))

    job: Mapped[IngestionJob] = relationship(back_populates="checkpoints")


class Artifact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("organization_id", "kind", "uri", name="artifact_uri_per_kind"),
        CheckConstraint("row_count >= 0", name="artifact_nonnegative_rows"),
        CheckConstraint("size_bytes >= 0", name="artifact_nonnegative_size"),
        Index("ix_artifacts_org_kind_status", "organization_id", "kind", "status"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    ingestion_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="SET NULL")
    )
    kind: Mapped[ArtifactKind] = mapped_column(
        enum_type(ArtifactKind, "artifact_kind"), nullable=False
    )
    status: Mapped[ArtifactStatus] = mapped_column(
        enum_type(ArtifactStatus, "artifact_status"),
        nullable=False,
        default=ArtifactStatus.STAGED,
    )
    uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str | None] = mapped_column(String(100))
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    min_date: Mapped[date | None] = mapped_column(Date)
    max_date: Mapped[date | None] = mapped_column(Date)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)

    ingestion_job: Mapped[IngestionJob | None] = relationship(back_populates="artifacts")


_FORBIDDEN_LABEL_KEYS = {
    "answer",
    "answer_key",
    "ground_truth",
    "insider",
    "insider_flag",
    "is_malicious",
    "is_threat",
    "label",
    "labels",
    "malicious",
    "scenario",
    "scenario_id",
}


def _normalise_payload_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _assert_label_free_payload(value: object) -> None:
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = _normalise_payload_key(raw_key)
            if (
                key in _FORBIDDEN_LABEL_KEYS
                or key.startswith("answer_")
                or key.startswith("label_")
                or key.startswith("scenario_")
            ):
                raise ValueError(f"Ground-truth key is forbidden in canonical data: {raw_key}")
            _assert_label_free_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_label_free_payload(nested)


class CanonicalEvent(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "canonical_events"
    __table_args__ = (
        UniqueConstraint("organization_id", "event_uid", name="canonical_event_uid_per_org"),
        UniqueConstraint(
            "organization_id",
            "source",
            "original_id",
            name="canonical_original_event_per_source",
        ),
        CheckConstraint("row_number IS NULL OR row_number > 0", name="event_positive_row_number"),
        Index(
            "ix_canonical_events_org_user_day_ts",
            "organization_id",
            "user_id",
            "event_date",
            "event_timestamp",
        ),
        Index(
            "ix_canonical_events_org_source_day",
            "organization_id",
            "source",
            "event_date",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    event_uid: Mapped[str] = mapped_column(String(160), nullable=False)
    source: Mapped[EventSource] = mapped_column(
        enum_type(EventSource, "event_source"), nullable=False
    )
    original_id: Mapped[str] = mapped_column(String(240), nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    pc: Mapped[str | None] = mapped_column(String(240))
    action: Mapped[str] = mapped_column(String(120), nullable=False)
    object_ref: Mapped[str | None] = mapped_column(Text)
    pc_context: Mapped[PCContext] = mapped_column(
        enum_type(PCContext, "pc_context"),
        nullable=False,
        default=PCContext.UNKNOWN,
    )
    legacy_time_context: Mapped[LegacyTimeContext] = mapped_column(
        "time_context",
        enum_type(LegacyTimeContext, "time_context"),
        nullable=False,
    )
    source_payload: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ingestion_job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="RESTRICT"), nullable=False
    )
    row_number: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    @validates("source_payload")
    def validate_source_payload(self, _key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("source_payload must be a JSON object")
        _assert_label_free_payload(payload)
        return payload


class FeatureCatalog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "feature_catalogs"
    __table_args__ = (
        UniqueConstraint("name", "version", name="feature_catalog_name_version"),
        UniqueConstraint("checksum", name="feature_catalog_checksum"),
        CheckConstraint("dimension > 0", name="feature_catalog_positive_dimension"),
    )

    name: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)

    definitions: Mapped[list[FeatureDefinition]] = relationship(
        back_populates="catalog",
        cascade="all, delete-orphan",
        order_by="FeatureDefinition.ordinal",
    )


class FeatureDefinition(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "feature_definitions"
    __table_args__ = (
        UniqueConstraint("catalog_id", "code", name="feature_code_per_catalog"),
        UniqueConstraint("catalog_id", "ordinal", name="feature_ordinal_per_catalog"),
        CheckConstraint("ordinal > 0", name="feature_positive_ordinal"),
    )

    catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("feature_catalogs.id", ondelete="CASCADE"), nullable=False
    )
    code: Mapped[str] = mapped_column(String(40), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    group_name: Mapped[str] = mapped_column(String(100), nullable=False)
    value_kind: Mapped[str] = mapped_column(String(80), nullable=False, default="numeric")
    source: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    constraints_json: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )

    catalog: Mapped[FeatureCatalog] = relationship(back_populates="definitions")


class UserDayFeature(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "user_day_features"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            "day",
            "catalog_id",
            name="feature_vector_per_user_day_catalog",
        ),
        CheckConstraint("feature_count > 0", name="feature_positive_count"),
        Index("ix_user_day_features_org_day", "organization_id", "day"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    role_assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("role_assignments.id", ondelete="SET NULL")
    )
    catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("feature_catalogs.id", ondelete="RESTRICT"), nullable=False
    )
    split: Mapped[DataSplit] = mapped_column(
        enum_type(DataSplit, "feature_data_split"), nullable=False
    )
    values: Mapped[list[float | None]] = mapped_column(json_type(), nullable=False)
    present_mask: Mapped[list[bool]] = mapped_column(json_type(), nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_observed_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_active_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    context_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)
    input_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL")
    )

    catalog: Mapped[FeatureCatalog] = relationship()


class UserDaySequence(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "user_day_sequences"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            "day",
            "vocabulary_version",
            name="sequence_per_user_day_vocabulary",
        ),
        CheckConstraint("seq_len >= 0", name="sequence_nonnegative_length"),
        CheckConstraint(
            "stored_len >= 0 AND stored_len <= 256",
            name="sequence_stored_length_range",
        ),
        Index("ix_user_day_sequences_org_day", "organization_id", "day"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    role_assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("role_assignments.id", ondelete="SET NULL")
    )
    split: Mapped[DataSplit] = mapped_column(
        enum_type(DataSplit, "sequence_data_split"), nullable=False
    )
    vocabulary_version: Mapped[str] = mapped_column(String(80), nullable=False)
    tokens: Mapped[list[str]] = mapped_column(json_type(), nullable=False)
    pc_contexts: Mapped[list[str]] = mapped_column(json_type(), nullable=False)
    # Keep the physical column name for compatibility with the initial schema;
    # sequence7.v4 stores only pipeline-derived WEEKDAY/WEEKEND values here.
    calendar_contexts: Mapped[list[str]] = mapped_column(
        "time_contexts", json_type(), nullable=False
    )
    gap_buckets: Mapped[list[str | None]] = mapped_column(json_type(), nullable=False)
    event_uids: Mapped[list[str]] = mapped_column(json_type(), nullable=False)
    side_fields: Mapped[list[dict[str, Any]]] = mapped_column(
        json_type(), nullable=False, default=list
    )
    seq_len: Mapped[int] = mapped_column(Integer, nullable=False)
    stored_len: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    input_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL")
    )


class ReferenceProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "reference_profiles"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "branch",
            "level",
            "scope_key",
            "model_version",
            "config_version",
            "fitted_through",
            name="reference_release_per_scope",
        ),
        CheckConstraint("support_days >= 0", name="reference_support_days_nonnegative"),
        CheckConstraint("support_users >= 0", name="reference_support_users_nonnegative"),
        CheckConstraint(
            "support_transitions >= 0",
            name="reference_support_transitions_nonnegative",
        ),
        CheckConstraint(
            "coverage IS NULL OR (coverage >= 0 AND coverage <= 1)",
            name="reference_coverage_range",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    branch: Mapped[Branch] = mapped_column(enum_type(Branch, "reference_branch"), nullable=False)
    level: Mapped[ReferenceLevel] = mapped_column(
        enum_type(ReferenceLevel, "reference_level"), nullable=False
    )
    scope_key: Mapped[str] = mapped_column(String(160), nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    role_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("roles.id", ondelete="RESTRICT"))
    role_assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("role_assignments.id", ondelete="RESTRICT")
    )
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    config_version: Mapped[str] = mapped_column(String(120), nullable=False)
    catalog_version: Mapped[str] = mapped_column(String(120), nullable=False)
    fitted_from: Mapped[date | None] = mapped_column(Date)
    fitted_through: Mapped[date] = mapped_column(Date, nullable=False)
    support_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    support_users: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    support_transitions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    coverage: Mapped[float | None] = mapped_column(Float)
    support_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)
    statistics_json: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    calibrator_json: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    is_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL")
    )


class BranchScore(Base):
    __tablename__ = "branch_scores"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            "day",
            "branch",
            "model_version",
            "config_version",
            name="branch_score_release_per_user_day",
        ),
        CheckConstraint(
            "(status = 'no_score' AND selected_level IS NULL "
            "AND reference_profile_id IS NULL AND raw_score IS NULL "
            "AND calibrated_score IS NULL) "
            "OR (status = 'scored' AND selected_level IS NOT NULL "
            "AND raw_score IS NOT NULL AND calibrated_score IS NOT NULL)",
            name="branch_score_status_shape",
        ),
        CheckConstraint(
            "calibrated_score IS NULL OR (calibrated_score >= 0 AND calibrated_score <= 1)",
            name="branch_calibrated_range",
        ),
        Index("ix_branch_scores_user_day", "user_id", "day"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    role_assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("role_assignments.id", ondelete="SET NULL")
    )
    branch: Mapped[Branch] = mapped_column(enum_type(Branch, "score_branch"), nullable=False)
    status: Mapped[ScoreStatus] = mapped_column(
        enum_type(ScoreStatus, "score_status"), nullable=False
    )
    selected_level: Mapped[ReferenceLevel | None] = mapped_column(
        enum_type(ReferenceLevel, "score_reference_level")
    )
    reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    raw_score: Mapped[float | None] = mapped_column(Float)
    calibrated_score: Mapped[float | None] = mapped_column(Float)
    support_snapshot: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    fallback_reasons: Mapped[list[str]] = mapped_column(json_type(), nullable=False, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    config_version: Mapped[str] = mapped_column(String(120), nullable=False)
    scoring_run_id: Mapped[str] = mapped_column(String(160), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class RiskAssessment(Base):
    """Immutable fused decision for a released model/config."""

    __tablename__ = "risk_assessments"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            "day",
            "model_version",
            "config_version",
            name="risk_release_per_user_day",
        ),
        CheckConstraint(
            "(status = 'no_score' AND feature_score_id IS NULL "
            "AND sequence_score_id IS NULL AND feature_weight = 0 "
            "AND sequence_weight = 0 AND risk IS NULL AND threshold IS NULL "
            "AND is_alert = false) "
            "OR (status = 'scored' AND "
            "(feature_score_id IS NOT NULL OR sequence_score_id IS NOT NULL) "
            "AND risk IS NOT NULL AND threshold IS NOT NULL)",
            name="risk_status_shape",
        ),
        CheckConstraint("risk IS NULL OR (risk >= 0 AND risk <= 1)", name="risk_value_range"),
        CheckConstraint(
            "threshold IS NULL OR (threshold >= 0 AND threshold <= 1)",
            name="risk_threshold_range",
        ),
        CheckConstraint(
            "status = 'no_score' OR "
            "(feature_weight + sequence_weight >= 0.999999 "
            "AND feature_weight + sequence_weight <= 1.000001)",
            name="risk_weights_sum",
        ),
        CheckConstraint(
            "status = 'no_score' OR "
            "(is_alert = true AND risk >= threshold) OR "
            "(is_alert = false AND risk < threshold)",
            name="risk_alert_matches_threshold",
        ),
        Index(
            "ix_risk_assessments_org_day_alert",
            "organization_id",
            "day",
            "is_alert",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    role_assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("role_assignments.id", ondelete="SET NULL")
    )
    split: Mapped[DataSplit] = mapped_column(
        enum_type(DataSplit, "assessment_data_split"), nullable=False
    )
    status: Mapped[AssessmentStatus] = mapped_column(
        enum_type(AssessmentStatus, "assessment_status"), nullable=False
    )
    feature_score_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("branch_scores.id", ondelete="RESTRICT")
    )
    sequence_score_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("branch_scores.id", ondelete="RESTRICT")
    )
    feature_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sequence_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    is_alert: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    config_version: Mapped[str] = mapped_column(String(120), nullable=False)
    scoring_run_id: Mapped[str] = mapped_column(String(160), nullable=False)
    fusion_evidence: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    feature_score: Mapped[BranchScore | None] = relationship(foreign_keys=[feature_score_id])
    sequence_score: Mapped[BranchScore | None] = relationship(foreign_keys=[sequence_score_id])
    alert: Mapped[Alert | None] = relationship(back_populates="assessment", uselist=False)


class Alert(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Mutable analyst workflow linked to an immutable assessment."""

    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("assessment_id", name="alert_per_assessment"),
        Index("ix_alerts_org_status_opened", "organization_id", "status", "opened_at"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("risk_assessments.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[AlertStatus] = mapped_column(
        enum_type(AlertStatus, "alert_status"),
        nullable=False,
        default=AlertStatus.OPEN,
    )
    severity: Mapped[AlertSeverity] = mapped_column(
        enum_type(AlertSeverity, "alert_severity"),
        nullable=False,
        default=AlertSeverity.MEDIUM,
    )
    assignee: Mapped[str | None] = mapped_column(String(240))
    resolution: Mapped[str | None] = mapped_column(Text)
    notes_json: Mapped[list[dict[str, Any]]] = mapped_column(
        json_type(), nullable=False, default=list
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment: Mapped[RiskAssessment] = relationship(back_populates="alert")


class SafeUpdateCandidate(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "safe_update_candidates"
    __table_args__ = (
        UniqueConstraint(
            "reference_profile_id",
            "branch",
            "candidate_day",
            name="safe_update_day_per_profile_branch",
        ),
        CheckConstraint(
            "quarantine_until >= candidate_day",
            name="safe_update_quarantine_after_day",
        ),
        CheckConstraint(
            "influence_cap >= 0 AND influence_cap <= 1",
            name="safe_update_influence_cap_range",
        ),
        Index(
            "ix_safe_update_candidates_status_quarantine",
            "status",
            "quarantine_until",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    role_assignment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("role_assignments.id", ondelete="RESTRICT"), nullable=False
    )
    reference_profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    source_assessment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("risk_assessments.id", ondelete="RESTRICT"), nullable=False
    )
    branch: Mapped[Branch] = mapped_column(enum_type(Branch, "safe_update_branch"), nullable=False)
    candidate_day: Mapped[date] = mapped_column(Date, nullable=False)
    quarantine_until: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[UpdateStatus] = mapped_column(
        enum_type(UpdateStatus, "safe_update_status"),
        nullable=False,
        default=UpdateStatus.CANDIDATE,
    )
    reason_codes: Mapped[list[str]] = mapped_column(json_type(), nullable=False, default=list)
    influence_cap: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    before_checksum: Mapped[str | None] = mapped_column(String(128))
    after_checksum: Mapped[str | None] = mapped_column(String(128))


class AuditLog(UUIDPrimaryKeyMixin, Base):
    """Append-only audit record."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        UniqueConstraint("event_hash", name="audit_event_hash"),
        Index("ix_audit_logs_entity_time", "entity_type", "entity_id", "occurred_at"),
        Index("ix_audit_logs_request", "request_id"),
    )

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT")
    )
    actor: Mapped[str] = mapped_column(String(240), nullable=False)
    action: Mapped[str] = mapped_column(String(120), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(120), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(160), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(160))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    before_hash: Mapped[str | None] = mapped_column(String(64))
    after_hash: Mapped[str | None] = mapped_column(String(64))
    previous_hash: Mapped[str | None] = mapped_column(String(64))
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ImmutableRecordError(RuntimeError):
    pass


def _column_state_changed(instance: object) -> bool:
    state = inspect(instance)
    return any(
        state.attrs[column.key].history.has_changes() for column in state.mapper.column_attrs
    )


def _dates_overlap(
    first_start: date,
    first_end: date | None,
    second_start: date,
    second_end: date | None,
) -> bool:
    upper = date.max
    return first_start < (second_end or upper) and second_start < (first_end or upper)


def _validate_feature_vector(session: Session, vector: UserDayFeature) -> None:
    if not isinstance(vector.values, list) or not isinstance(vector.present_mask, list):
        raise ValueError("Feature values and present_mask must be JSON arrays")
    if len(vector.values) != len(vector.present_mask):
        raise ValueError("Feature values and present_mask must have equal length")
    for value, present in zip(vector.values, vector.present_mask, strict=True):
        if value is None:
            if present:
                raise ValueError("Undefined feature values require present_mask=false")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Every present feature value must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("Feature values cannot contain NaN or infinity")
    catalog = vector.catalog or session.get(FeatureCatalog, vector.catalog_id)
    if catalog is None:
        raise ValueError("Feature vector requires an existing catalog")
    if len(vector.values) != catalog.dimension:
        raise ValueError(
            f"Feature vector has {len(vector.values)} values; catalog requires {catalog.dimension}"
        )
    vector.feature_count = len(vector.values)


def _validate_sequence(sequence: UserDaySequence) -> None:
    arrays = [
        sequence.tokens,
        sequence.pc_contexts,
        sequence.calendar_contexts,
        sequence.gap_buckets,
        sequence.event_uids,
    ]
    if any(not isinstance(values, list) for values in arrays):
        raise ValueError("Sequence fields must be JSON arrays")
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError("All sequence side-channel arrays must have equal length")
    stored_len = lengths.pop()
    if sequence.side_fields and len(sequence.side_fields) != stored_len:
        raise ValueError("Sequence side_fields must align with tokens")
    if stored_len > 256:
        raise ValueError("Stored sequence cannot exceed max_len=256")
    if sequence.seq_len < stored_len:
        raise ValueError("seq_len cannot be shorter than stored sequence")
    if sequence.seq_len > 256:
        if stored_len != 256 or not sequence.truncated:
            raise ValueError("Long sequences must store 256 events and set truncated=true")
    elif stored_len != sequence.seq_len or sequence.truncated:
        raise ValueError("Untruncated sequence must store exactly seq_len events")
    sequence.stored_len = stored_len


def _validate_reference_scope(profile: ReferenceProfile) -> None:
    if profile.level is ReferenceLevel.PERSON:
        if (
            profile.user_id is None
            or profile.role_assignment_id is None
            or profile.role_id is not None
        ):
            raise ValueError("PERSON reference requires user and role assignment only")
        expected = f"person:{profile.role_assignment_id}"
    elif profile.level is ReferenceLevel.ROLE:
        if (
            profile.role_id is None
            or profile.user_id is not None
            or profile.role_assignment_id is not None
        ):
            raise ValueError("ROLE reference requires role only")
        expected = f"role:{profile.role_id}"
    else:
        if any(
            value is not None
            for value in (
                profile.user_id,
                profile.role_id,
                profile.role_assignment_id,
            )
        ):
            raise ValueError("GLOBAL reference cannot have user/role scope IDs")
        expected = "global"
    if not profile.scope_key:
        profile.scope_key = expected
    elif profile.scope_key != expected:
        raise ValueError(f"Reference scope_key must be {expected!r}, got {profile.scope_key!r}")


def _validate_assessment_branches(session: Session, assessment: RiskAssessment) -> None:
    if assessment.status is AssessmentStatus.NO_SCORE:
        return
    for score_id, branch in (
        (assessment.feature_score_id, Branch.FEATURE),
        (assessment.sequence_score_id, Branch.SEQUENCE),
    ):
        if score_id is None:
            continue
        score = session.get(BranchScore, score_id)
        if score is None or score.branch is not branch:
            raise ValueError(f"Risk assessment has an invalid {branch.value} score")
        if score.status is not ScoreStatus.SCORED:
            raise ValueError("Risk assessment cannot fuse a NO_SCORE branch")
        if score.user_id != assessment.user_id or score.day != assessment.day:
            raise ValueError("Risk assessment scores must belong to the same user-day")


@event.listens_for(Session, "before_flush")
def _enforce_persistence_invariants(
    session: Session, _flush_context: object, _instances: object
) -> None:
    immutable_types = (RiskAssessment, AuditLog)
    for instance in session.deleted:
        if isinstance(instance, immutable_types):
            raise ImmutableRecordError(f"{type(instance).__name__} records cannot be deleted")
    for instance in session.dirty:
        if isinstance(instance, immutable_types) and _column_state_changed(instance):
            raise ImmutableRecordError(f"{type(instance).__name__} records cannot be updated")

    pending = [
        item
        for item in session.new.union(session.dirty)
        if isinstance(item, RoleAssignment) and (item in session.new or _column_state_changed(item))
    ]
    for index, assignment in enumerate(pending):
        if assignment.user_id is None or assignment.valid_from is None:
            continue
        for other in pending[index + 1 :]:
            if (
                other.user_id == assignment.user_id
                and other.valid_from is not None
                and _dates_overlap(
                    assignment.valid_from,
                    assignment.valid_to,
                    other.valid_from,
                    other.valid_to,
                )
            ):
                raise ValueError("Role assignments for one user cannot overlap")
        with session.no_autoflush:
            query = select(RoleAssignment.id).where(
                RoleAssignment.user_id == assignment.user_id,
                RoleAssignment.valid_from < (assignment.valid_to or date.max),
                or_(
                    RoleAssignment.valid_to.is_(None),
                    RoleAssignment.valid_to > assignment.valid_from,
                ),
            )
            if assignment.id is not None:
                query = query.where(RoleAssignment.id != assignment.id)
            if session.scalar(query.limit(1)) is not None:
                raise ValueError("Role assignments for one user cannot overlap")

    for instance in session.new.union(session.dirty):
        if instance not in session.new and not _column_state_changed(instance):
            continue
        if isinstance(instance, UserDayFeature):
            _validate_feature_vector(session, instance)
        elif isinstance(instance, UserDaySequence):
            _validate_sequence(instance)
        elif isinstance(instance, ReferenceProfile):
            _validate_reference_scope(instance)
        elif isinstance(instance, RiskAssessment):
            _validate_assessment_branches(session, instance)


Person = User
