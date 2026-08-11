"""Portable SQLAlchemy models for the operational insider-threat database.

Ground-truth labels intentionally do not exist in this schema. Evaluation data
belongs to the physically separate database described in ``data/evaluation/``.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import UTC, date, datetime, timedelta
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


class PersonalAccumulatorStatus(StrEnum):
    WARMING = "warming"
    ACTIVE = "active"
    CLOSED = "closed"
    COMPROMISED = "compromised"


class ReferenceReleaseKind(StrEnum):
    LEGACY = "legacy"
    BOOTSTRAP = "bootstrap"
    INCREMENTAL = "incremental"


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
    original_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    pc: Mapped[str | None] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(128), nullable=False)
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
        CheckConstraint(
            "reference_version IS NULL OR reference_version > 0",
            name="reference_version_positive",
        ),
        CheckConstraint(
            "release_influence_ratio IS NULL OR "
            "(release_influence_ratio >= 0 AND release_influence_ratio <= 1)",
            name="reference_release_influence_range",
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
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_version: Mapped[str] = mapped_column(String(128), nullable=False)
    catalog_version: Mapped[str] = mapped_column(String(128), nullable=False)
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
    parent_reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    calibration_parent_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    reference_version: Mapped[int | None] = mapped_column(Integer)
    release_kind: Mapped[ReferenceReleaseKind | None] = mapped_column(
        enum_type(ReferenceReleaseKind, "reference_profile_release_kind")
    )
    release_day: Mapped[date | None] = mapped_column(Date)
    release_influence_ratio: Mapped[float | None] = mapped_column(Float)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
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
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_version: Mapped[str] = mapped_column(String(128), nullable=False)
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
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_version: Mapped[str] = mapped_column(String(128), nullable=False)
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
    assignee: Mapped[str | None] = mapped_column(String(255))
    resolution: Mapped[str | None] = mapped_column(Text)
    notes_json: Mapped[list[dict[str, Any]]] = mapped_column(
        json_type(), nullable=False, default=list
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment: Mapped[RiskAssessment] = relationship(back_populates="alert")


class PersonalReferenceAccumulator(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Mutable pointer and release counter for one Personal branch/role epoch."""

    __tablename__ = "personal_reference_accumulators"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            "role_assignment_id",
            "branch",
            "model_version",
            "config_version",
            "catalog_version",
            name="personal_accumulator_scope_release",
        ),
        CheckConstraint(
            "release_sequence >= 0",
            name="personal_accumulator_release_sequence_nonnegative",
        ),
        CheckConstraint(
            "status != 'active' OR active_reference_profile_id IS NOT NULL",
            name="personal_accumulator_active_reference_required",
        ),
        Index(
            "ix_personal_accumulators_org_status",
            "organization_id",
            "status",
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
    branch: Mapped[Branch] = mapped_column(
        enum_type(Branch, "personal_accumulator_branch"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_version: Mapped[str] = mapped_column(String(128), nullable=False)
    catalog_version: Mapped[str] = mapped_column(String(128), nullable=False)
    admission_reference_profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    active_reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    status: Mapped[PersonalAccumulatorStatus] = mapped_column(
        enum_type(PersonalAccumulatorStatus, "personal_accumulator_status"),
        nullable=False,
        default=PersonalAccumulatorStatus.WARMING,
    )
    release_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_release_day: Mapped[date | None] = mapped_column(Date)


class ScoringWatermark(UUIDPrimaryKeyMixin, Base):
    """Immutable proof that the production scoring universe is complete for a day."""

    __tablename__ = "scoring_watermarks"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "day",
            "model_version",
            "config_version",
            name="scoring_watermark_release",
        ),
        CheckConstraint(
            "expected_assessments >= 0",
            name="scoring_watermark_expected_nonnegative",
        ),
        CheckConstraint(
            "persisted_assessments = expected_assessments",
            name="scoring_watermark_complete",
        ),
        Index(
            "ix_scoring_watermarks_org_day",
            "organization_id",
            "day",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_version: Mapped[str] = mapped_column(String(128), nullable=False)
    expected_assessments: Mapped[int] = mapped_column(Integer, nullable=False)
    persisted_assessments: Mapped[int] = mapped_column(Integer, nullable=False)
    universe_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    assessment_set_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ReferenceRelease(UUIDPrimaryKeyMixin, Base):
    """Immutable manifest linking accepted candidates to a Personal reference."""

    __tablename__ = "reference_releases"
    __table_args__ = (
        UniqueConstraint(
            "accumulator_id",
            "release_sequence",
            name="reference_release_sequence",
        ),
        UniqueConstraint(
            "child_reference_profile_id",
            name="reference_release_child_profile",
        ),
        UniqueConstraint("manifest_checksum", name="reference_release_manifest_checksum"),
        CheckConstraint(
            "release_sequence > 0",
            name="reference_release_sequence_positive",
        ),
        CheckConstraint(
            "parent_support >= 0 AND applied_candidate_count > 0",
            name="reference_release_support_positive",
        ),
        CheckConstraint(
            "influence_ratio >= 0 AND influence_ratio <= 1",
            name="reference_release_influence_range",
        ),
        CheckConstraint(
            "rolling_anchor_support >= 0 AND rolling_applied_count_before >= 0",
            name="reference_release_rolling_nonnegative",
        ),
        CheckConstraint(
            "(kind = 'bootstrap' AND parent_reference_profile_id IS NULL) OR "
            "(kind = 'incremental' AND parent_reference_profile_id IS NOT NULL)",
            name="reference_release_parent_shape",
        ),
        CheckConstraint(
            "kind IN ('bootstrap', 'incremental')",
            name="reference_release_materialized_kind",
        ),
        Index(
            "ix_reference_releases_accumulator_day",
            "accumulator_id",
            "release_day",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False
    )
    accumulator_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("personal_reference_accumulators.id", ondelete="RESTRICT"),
        nullable=False,
    )
    release_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[ReferenceReleaseKind] = mapped_column(
        enum_type(ReferenceReleaseKind, "reference_release_kind"), nullable=False
    )
    parent_reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    calibration_parent_profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    child_reference_profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    release_day: Mapped[date] = mapped_column(Date, nullable=False)
    parent_support: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    influence_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    rolling_anchor_support: Mapped[int] = mapped_column(Integer, nullable=False)
    rolling_applied_count_before: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_manifest_json: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False
    )
    manifest_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    released_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class SafeUpdateCandidate(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "safe_update_candidates"
    __table_args__ = (
        UniqueConstraint(
            "reference_profile_id",
            "branch",
            "candidate_day",
            name="safe_update_day_per_profile_branch",
        ),
        UniqueConstraint(
            "accumulator_id",
            "candidate_day",
            name="safe_update_day_per_accumulator",
        ),
        UniqueConstraint(
            "source_branch_score_id",
            name="safe_update_source_branch_score",
        ),
        CheckConstraint(
            "quarantine_until >= candidate_day",
            name="safe_update_quarantine_after_day",
        ),
        CheckConstraint(
            "influence_cap >= 0 AND influence_cap <= 1",
            name="safe_update_influence_cap_range",
        ),
        CheckConstraint(
            "admission_percentile IS NULL OR "
            "(admission_percentile >= 0 AND admission_percentile <= 1)",
            name="safe_update_admission_percentile_range",
        ),
        CheckConstraint(
            "admission_threshold IS NULL OR "
            "(admission_threshold > 0 AND admission_threshold < 1)",
            name="safe_update_admission_threshold_range",
        ),
        CheckConstraint(
            "eligible_on IS NULL OR eligible_on >= quarantine_until",
            name="safe_update_eligible_after_quarantine",
        ),
        Index(
            "ix_safe_update_candidates_status_quarantine",
            "status",
            "quarantine_until",
        ),
        Index(
            "ix_safe_update_candidates_policy_eligible",
            "policy_version",
            "status",
            "eligible_on",
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
    reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    source_assessment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("risk_assessments.id", ondelete="RESTRICT"), nullable=False
    )
    accumulator_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("personal_reference_accumulators.id", ondelete="RESTRICT")
    )
    source_branch_score_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("branch_scores.id", ondelete="RESTRICT")
    )
    source_feature_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("user_day_features.id", ondelete="RESTRICT")
    )
    source_sequence_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("user_day_sequences.id", ondelete="RESTRICT")
    )
    source_input_checksum: Mapped[str | None] = mapped_column(String(128))
    admission_reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    admission_reference_checksum: Mapped[str | None] = mapped_column(String(128))
    admission_percentile: Mapped[float | None] = mapped_column(Float)
    admission_threshold: Mapped[float | None] = mapped_column(Float)
    branch: Mapped[Branch] = mapped_column(enum_type(Branch, "safe_update_branch"), nullable=False)
    candidate_day: Mapped[date] = mapped_column(Date, nullable=False)
    quarantine_until: Mapped[date] = mapped_column(Date, nullable=False)
    eligible_on: Mapped[date | None] = mapped_column(Date)
    status: Mapped[UpdateStatus] = mapped_column(
        enum_type(UpdateStatus, "safe_update_status"),
        nullable=False,
        default=UpdateStatus.CANDIDATE,
    )
    reason_codes: Mapped[list[str]] = mapped_column(json_type(), nullable=False, default=list)
    support_contribution_json: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(128))
    config_version: Mapped[str | None] = mapped_column(String(128))
    influence_cap: Mapped[float] = mapped_column(Float, nullable=False, default=0.02)
    decision_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    before_checksum: Mapped[str | None] = mapped_column(String(128))
    after_checksum: Mapped[str | None] = mapped_column(String(128))
    materialized_reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="RESTRICT")
    )
    reference_release_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reference_releases.id", ondelete="RESTRICT")
    )


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


def _changed_column_names(instance: object) -> set[str]:
    state = inspect(instance)
    return {
        column.key
        for column in state.mapper.column_attrs
        if state.attrs[column.key].history.has_changes()
    }


def _previous_column_value(instance: object, name: str) -> Any:
    history = inspect(instance).attrs[name].history
    if history.deleted:
        return history.deleted[0]
    if history.unchanged:
        return history.unchanged[0]
    raise ImmutableRecordError(
        f"Cannot validate {type(instance).__name__}.{name}: prior value is unavailable"
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


def _validate_reference_lineage(session: Session, profile: ReferenceProfile) -> None:
    if not profile.policy_version:
        profile.policy_version = profile.config_version
    if profile.reference_version is not None and profile.reference_version < 1:
        raise ValueError("Reference version must be positive")
    if profile.release_influence_ratio is not None and not (
        0 <= profile.release_influence_ratio <= 1
    ):
        raise ValueError("Reference release influence ratio must be in [0,1]")
    if profile.release_kind is ReferenceReleaseKind.LEGACY:
        if (
            profile.config_version != "framework.v4"
            or profile.policy_version != "framework.v4"
        ):
            raise ValueError("Legacy reference lineage is restricted to framework.v4 profiles")
        if any(
            value is not None
            for value in (
                profile.parent_reference_profile_id,
                profile.calibration_parent_profile_id,
                profile.reference_version,
                profile.release_day,
                profile.release_influence_ratio,
            )
        ):
            raise ValueError("Legacy references cannot claim v5 release lineage")
        return
    if profile.release_kind is None:
        if any(
            value is not None
            for value in (
                profile.parent_reference_profile_id,
                profile.calibration_parent_profile_id,
                profile.release_day,
                profile.release_influence_ratio,
            )
        ):
            raise ValueError("Reference lineage fields require release_kind")
        return
    if (
        profile.level is not ReferenceLevel.PERSON
        or profile.reference_version is None
        or profile.release_day is None
        or profile.release_influence_ratio is None
        or profile.calibration_parent_profile_id is None
    ):
        raise ValueError("Released references require complete Personal lineage")
    if profile.release_kind is ReferenceReleaseKind.BOOTSTRAP:
        if profile.parent_reference_profile_id is not None or profile.reference_version != 1:
            raise ValueError("Bootstrap reference must be version 1 without a Personal parent")
    elif profile.parent_reference_profile_id is None or profile.reference_version <= 1:
        raise ValueError("Incremental reference requires a prior Personal version")

    parent_ids = {
        profile.parent_reference_profile_id,
        profile.calibration_parent_profile_id,
    } - {None}
    for parent_id in parent_ids:
        parent = session.get(ReferenceProfile, parent_id)
        if parent is None:
            raise ValueError("Reference lineage points to a missing profile")
        if (
            parent.organization_id != profile.organization_id
            or parent.branch is not profile.branch
            or parent.model_version != profile.model_version
            or parent.config_version != profile.config_version
            or parent.catalog_version != profile.catalog_version
        ):
            raise ValueError("Reference lineage must remain in one release scope")
    calibration_parent = session.get(
        ReferenceProfile,
        profile.calibration_parent_profile_id,
    )
    if calibration_parent is None or calibration_parent.level is ReferenceLevel.PERSON:
        raise ValueError("Calibration parent must be a ROLE or GLOBAL reference")
    if not calibration_parent.is_frozen:
        raise ValueError("Calibration parent must be frozen")


def _validate_accumulator(
    session: Session,
    accumulator: PersonalReferenceAccumulator,
) -> None:
    assignment = session.get(RoleAssignment, accumulator.role_assignment_id)
    admission = session.get(ReferenceProfile, accumulator.admission_reference_profile_id)
    if assignment is None or assignment.user_id != accumulator.user_id:
        raise ValueError("Personal accumulator requires the user's role epoch anchor")
    if admission is None or admission.level is ReferenceLevel.PERSON:
        raise ValueError("Personal accumulator admission reference must be ROLE or GLOBAL")
    if (
        admission.organization_id != accumulator.organization_id
        or admission.branch is not accumulator.branch
        or admission.model_version != accumulator.model_version
        or admission.config_version != accumulator.config_version
        or admission.catalog_version != accumulator.catalog_version
    ):
        raise ValueError("Personal accumulator and admission reference releases must match")
    if admission.level is ReferenceLevel.ROLE and admission.role_id != assignment.role_id:
        raise ValueError("ROLE admission reference must match the accumulator role epoch")
    if not admission.is_frozen:
        raise ValueError("Personal accumulator admission reference must be frozen")
    if accumulator.active_reference_profile_id is None:
        if accumulator.status is PersonalAccumulatorStatus.ACTIVE:
            raise ValueError("Active Personal accumulator requires an active reference")
        return
    active = session.get(ReferenceProfile, accumulator.active_reference_profile_id)
    if active is None or active.level is not ReferenceLevel.PERSON:
        raise ValueError("Accumulator active reference must be a Personal profile")
    if (
        active.organization_id != accumulator.organization_id
        or active.user_id != accumulator.user_id
        or active.role_assignment_id != accumulator.role_assignment_id
        or active.branch is not accumulator.branch
        or active.model_version != accumulator.model_version
        or active.config_version != accumulator.config_version
        or active.catalog_version != accumulator.catalog_version
    ):
        raise ValueError("Accumulator active reference must match its release scope")


def _validate_reference_release(session: Session, release: ReferenceRelease) -> None:
    if release.kind not in (
        ReferenceReleaseKind.BOOTSTRAP,
        ReferenceReleaseKind.INCREMENTAL,
    ):
        raise ValueError("Legacy references cannot be materialized release events")
    accumulator = session.get(PersonalReferenceAccumulator, release.accumulator_id)
    child = session.get(ReferenceProfile, release.child_reference_profile_id)
    calibration_parent = session.get(
        ReferenceProfile,
        release.calibration_parent_profile_id,
    )
    if accumulator is None or child is None or calibration_parent is None:
        raise ValueError("Reference release requires existing accumulator and profiles")
    if release.organization_id != accumulator.organization_id:
        raise ValueError("Reference release organization must match its accumulator")
    if (
        child.level is not ReferenceLevel.PERSON
        or child.user_id != accumulator.user_id
        or child.role_assignment_id != accumulator.role_assignment_id
        or child.branch is not accumulator.branch
        or child.model_version != accumulator.model_version
        or child.config_version != accumulator.config_version
        or child.catalog_version != accumulator.catalog_version
        or child.release_kind is not release.kind
        or child.reference_version != release.release_sequence
        or child.release_day != release.release_day
        or child.calibration_parent_profile_id != release.calibration_parent_profile_id
    ):
        raise ValueError("Reference release child does not match its accumulator lineage")
    if release.kind is ReferenceReleaseKind.INCREMENTAL:
        if child.parent_reference_profile_id != release.parent_reference_profile_id:
            raise ValueError("Incremental release parent does not match child lineage")
        parent = session.get(ReferenceProfile, release.parent_reference_profile_id)
        if parent is None or parent.level is not ReferenceLevel.PERSON:
            raise ValueError("Incremental release requires a Personal parent")
    if calibration_parent.id != accumulator.admission_reference_profile_id:
        raise ValueError("Release calibration parent must be the immutable admission anchor")
    candidate_ids = release.candidate_manifest_json.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != release.applied_candidate_count
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(not isinstance(value, str) or not value for value in candidate_ids)
    ):
        raise ValueError("Reference release manifest must contain unique candidate IDs")


def _validate_safe_update_candidate(
    session: Session,
    candidate: SafeUpdateCandidate,
) -> None:
    assessment = session.get(RiskAssessment, candidate.source_assessment_id)
    if assessment is None:
        raise ValueError("Safe-update candidate requires a source assessment")
    if not candidate.model_version:
        candidate.model_version = assessment.model_version
    if not candidate.config_version:
        candidate.config_version = assessment.config_version
    if not candidate.policy_version:
        candidate.policy_version = (
            "framework.v5.safe_update.v1"
            if candidate.config_version == "framework.v5"
            else "framework.v4.safe_update.v1"
        )
    if candidate.eligible_on is None:
        candidate.eligible_on = candidate.quarantine_until
    if candidate.policy_version != "framework.v5.safe_update.v1":
        return

    required = {
        "accumulator_id": candidate.accumulator_id,
        "source_branch_score_id": candidate.source_branch_score_id,
        "source_input_checksum": candidate.source_input_checksum,
        "admission_reference_profile_id": candidate.admission_reference_profile_id,
        "admission_reference_checksum": candidate.admission_reference_checksum,
        "admission_percentile": candidate.admission_percentile,
        "admission_threshold": candidate.admission_threshold,
        "model_version": candidate.model_version,
        "config_version": candidate.config_version,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"v5 safe-update candidate lacks evidence: {', '.join(missing)}")
    if candidate.config_version != "framework.v5":
        raise ValueError("v5 safe-update policy requires framework.v5 config")
    if candidate.quarantine_until != candidate.candidate_day + timedelta(days=30):
        raise ValueError("v5 quarantine must include D through D+30")
    if candidate.eligible_on != candidate.candidate_day + timedelta(days=31):
        raise ValueError("v5 candidate is first eligible on D+31")
    if candidate.admission_percentile >= candidate.admission_threshold:
        raise ValueError("v5 admission percentile must be strictly below its threshold")
    if candidate.influence_cap > 0.02:
        raise ValueError("v5 candidate influence cap cannot exceed 2%")
    if assessment.split is not DataSplit.PRODUCTION:
        raise ValueError("Safe-update candidate must originate in PRODUCTION")
    if (
        assessment.status is not AssessmentStatus.SCORED
        or assessment.is_alert
        or assessment.user_id != candidate.user_id
        or assessment.day != candidate.candidate_day
        or assessment.organization_id != candidate.organization_id
        or assessment.model_version != candidate.model_version
        or assessment.config_version != candidate.config_version
    ):
        raise ValueError("Safe-update source assessment is not an admissible user-day")

    accumulator = session.get(PersonalReferenceAccumulator, candidate.accumulator_id)
    admission = session.get(ReferenceProfile, candidate.admission_reference_profile_id)
    score = session.get(BranchScore, candidate.source_branch_score_id)
    if accumulator is None or admission is None or score is None:
        raise ValueError("v5 safe-update candidate points to missing evidence")
    if (
        accumulator.organization_id != candidate.organization_id
        or accumulator.user_id != candidate.user_id
        or accumulator.role_assignment_id != candidate.role_assignment_id
        or accumulator.branch is not candidate.branch
        or accumulator.model_version != candidate.model_version
        or accumulator.config_version != candidate.config_version
        or accumulator.admission_reference_profile_id != admission.id
    ):
        raise ValueError("Safe-update candidate does not match its accumulator")
    if (
        admission.level is ReferenceLevel.PERSON
        or not admission.is_frozen
        or admission.branch is not candidate.branch
        or admission.checksum != candidate.admission_reference_checksum
    ):
        raise ValueError("Safe-update admission reference evidence is invalid")
    expected_score_id = (
        assessment.feature_score_id
        if candidate.branch is Branch.FEATURE
        else assessment.sequence_score_id
    )
    if (
        score.id != expected_score_id
        or score.status is not ScoreStatus.SCORED
        or score.organization_id != candidate.organization_id
        or score.user_id != candidate.user_id
        or score.day != candidate.candidate_day
        or score.branch is not candidate.branch
        or score.model_version != candidate.model_version
        or score.config_version != candidate.config_version
        or score.raw_score is None
        or not math.isfinite(score.raw_score)
    ):
        raise ValueError("Safe-update source branch score evidence is invalid")
    if candidate.branch is Branch.FEATURE:
        if candidate.source_feature_id is None or candidate.source_sequence_id is not None:
            raise ValueError("Feature candidate requires exactly one feature input")
        source_input = session.get(UserDayFeature, candidate.source_feature_id)
    else:
        if candidate.source_sequence_id is None or candidate.source_feature_id is not None:
            raise ValueError("Sequence candidate requires exactly one sequence input")
        source_input = session.get(UserDaySequence, candidate.source_sequence_id)
    if (
        source_input is None
        or source_input.organization_id != candidate.organization_id
        or source_input.user_id != candidate.user_id
        or source_input.day != candidate.candidate_day
    ):
        raise ValueError("Safe-update source input evidence is invalid")
    checksum_changed = source_input.input_checksum != candidate.source_input_checksum
    rejected_for_changed_input = (
        candidate.status is UpdateStatus.REJECTED
        and "SOURCE_USER_DAY_CHANGED" in candidate.reason_codes
    )
    if checksum_changed and not rejected_for_changed_input:
        raise ValueError("Safe-update source input checksum no longer matches")
    if candidate.status is UpdateStatus.APPLIED:
        if any(
            value is None
            for value in (
                candidate.decision_at,
                candidate.applied_at,
                candidate.before_checksum,
                candidate.after_checksum,
                candidate.materialized_reference_profile_id,
                candidate.reference_release_id,
            )
        ):
            raise ValueError("Applied v5 candidate requires complete release evidence")
    elif (
        candidate.materialized_reference_profile_id is not None
        or candidate.reference_release_id is not None
    ):
        raise ValueError("Only an applied candidate may point to a materialized release")


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
    immutable_types = (
        ReferenceProfile,
        BranchScore,
        RiskAssessment,
        ScoringWatermark,
        ReferenceRelease,
        AuditLog,
    )
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
            _validate_reference_lineage(session, instance)
        elif isinstance(instance, PersonalReferenceAccumulator):
            _validate_accumulator(session, instance)
        elif isinstance(instance, ReferenceRelease):
            _validate_reference_release(session, instance)
        elif isinstance(instance, SafeUpdateCandidate):
            _validate_safe_update_candidate(session, instance)
        elif isinstance(instance, RiskAssessment):
            _validate_assessment_branches(session, instance)


Person = User
