from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceType(StrEnum):
    LOGON = "LOGON"
    DEVICE = "DEVICE"
    FILE = "FILE"
    HTTP = "HTTP"
    EMAIL = "EMAIL"


class IngestSourceType(StrEnum):
    LOGON = "LOGON"
    DEVICE = "DEVICE"
    FILE = "FILE"
    HTTP = "HTTP"
    EMAIL = "EMAIL"
    LDAP = "LDAP"


class PcContext(StrEnum):
    OWN = "OWN"
    SHARED = "SHARED"
    FOREIGN = "FOREIGN"
    UNKNOWN = "UNKNOWN"


class TimestampBasis(StrEnum):
    CERT_LOCAL_WALL_CLOCK = "CERT_LOCAL_WALL_CLOCK"
    ABSOLUTE_OFFSET = "ABSOLUTE_OFFSET"


class SequenceToken(StrEnum):
    LOGON = "LOGON"
    LOGOFF = "LOGOFF"
    DEVICE_CONNECT = "DEVICE_CONNECT"
    DEVICE_DISCONNECT = "DEVICE_DISCONNECT"
    FILE = "FILE"
    HTTP = "HTTP"
    EMAIL = "EMAIL"


class Branch(StrEnum):
    FEATURE = "FEATURE"
    SEQUENCE = "SEQUENCE"


class ReferenceLevel(StrEnum):
    PERSON = "PERSON"
    ROLE = "ROLE"
    GLOBAL = "GLOBAL"
    NO_SCORE = "NO_SCORE"


class AlertStatusFilter(StrEnum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_REVIEW = "IN_REVIEW"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    DISMISSED = "DISMISSED"
    FALSE_POSITIVE = "FALSE_POSITIVE"

    @classmethod
    def _missing_(cls, value: object) -> AlertStatusFilter | None:
        if isinstance(value, str):
            return cls.__members__.get(value.upper())
        return None


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserCreate(StrictModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$")
    display_name: str | None = Field(default=None, max_length=255)
    status: str = Field(default="ACTIVE", pattern=r"^(ACTIVE|INACTIVE)$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserRead(ORMModel):
    user_id: str
    display_name: str | None
    status: str
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class RoleCreate(StrictModel):
    role_code: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$")
    display_name: str = Field(min_length=1, max_length=255)
    role_family: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoleRead(ORMModel):
    role_code: str
    display_name: str
    role_family: str | None
    metadata_json: dict[str, Any]
    created_at: datetime


class RoleAssignmentCreate(StrictModel):
    role_code: str = Field(min_length=1, max_length=128)
    valid_from: date
    valid_to: date | None = None
    source_snapshot_date: date

    @model_validator(mode="after")
    def validate_dates(self) -> RoleAssignmentCreate:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be later than valid_from; intervals are [from, to)")
        if self.source_snapshot_date > self.valid_from:
            raise ValueError("future LDAP snapshots cannot be backfilled into an earlier role date")
        return self


class RoleAssignmentRead(ORMModel):
    id: str
    user_id: str
    role_code: str
    valid_from: date
    valid_to: date | None
    source_snapshot_date: date
    created_at: datetime


class IngestionJobCreate(StrictModel):
    source: IngestSourceType
    input_uri: str = Field(min_length=1, max_length=1024)
    original_filename: str | None = Field(default=None, max_length=300)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    idempotency_key: str = Field(min_length=1, max_length=160)
    schema_version: str | None = Field(default=None, max_length=80)
    total_rows: int = Field(default=0, ge=0)


class IngestionJobRead(ORMModel):
    id: str
    source: str
    status: str
    input_uri: str
    original_filename: str | None
    sha256: str
    idempotency_key: str
    schema_version: str | None
    total_rows: int
    processed_rows: int
    rejected_rows: int
    progress_json: dict[str, Any]
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CheckpointUpsert(StrictModel):
    partition_key: str = Field(
        min_length=1,
        max_length=240,
        description="Stable source/month key, for example HTTP/2010-03",
    )
    row_offset: int = Field(ge=0)
    cursor: dict[str, Any] = Field(default_factory=dict)
    input_checksum: str | None = Field(default=None, max_length=64)
    output_checksum: str | None = Field(default=None, max_length=64)


class CheckpointRead(ORMModel):
    id: str
    job_id: str
    partition_key: str
    row_offset: int
    cursor_json: dict[str, Any]
    input_checksum: str | None
    output_checksum: str | None
    created_at: datetime
    updated_at: datetime


class CanonicalEventIn(StrictModel):
    event_uid: str = Field(min_length=1, max_length=160)
    source: SourceType
    original_id: str = Field(min_length=1, max_length=255)
    timestamp: datetime
    timestamp_basis: TimestampBasis = TimestampBasis.CERT_LOCAL_WALL_CLOCK
    user_id: str = Field(min_length=1, max_length=128)
    pc: str | None = Field(default=None, max_length=255)
    action: str = Field(min_length=1, max_length=128)
    object: str | None = Field(default=None, max_length=2048)
    source_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_source_semantics(self) -> CanonicalEventIn:
        if (
            self.timestamp_basis is TimestampBasis.ABSOLUTE_OFFSET
            and (self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None)
        ):
            raise ValueError("ABSOLUTE_OFFSET timestamp must include a timezone offset")
        action = self.action.upper()
        if self.source is SourceType.FILE and action not in {"FILE_COPY", "COPY"}:
            raise ValueError("CERT file events mean removable-media copy; use FILE_COPY")
        if self.source is SourceType.HTTP:
            forbidden_claims = ("UPLOAD", "DOWNLOAD", "RESPONSE_STATUS", "HTTP_METHOD")
            if any(claim in action for claim in forbidden_claims):
                raise ValueError(
                    "CERT HTTP data cannot support upload/download/method/status claims"
                )
        return self


class EventBatchCreate(StrictModel):
    ingestion_job_id: str | None = None
    events: list[CanonicalEventIn] = Field(min_length=1, max_length=1000)


class EventBatchResult(BaseModel):
    inserted: int
    duplicates: int
    event_uids: list[str]


class EventRead(ORMModel):
    event_uid: str
    source: str
    original_id: str
    timestamp: datetime
    event_date: date
    user_id: str
    pc: str | None
    action: str
    object_ref: str | None
    pc_context: str
    source_payload_json: dict[str, Any]
    ingestion_job_id: str | None
    created_at: datetime


class FeatureVectorUpsert(StrictModel):
    schema_version: str = Field(default="feature128.v5", min_length=1, max_length=64)
    values: dict[str, float | None]
    masks: dict[str, bool] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    input_checksum: str | None = Field(default=None, max_length=128)

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, values: dict[str, float | None]) -> dict[str, float | None]:
        for name, value in values.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return values


class FeatureVectorRead(ORMModel):
    id: str
    user_id: str
    day: date
    schema_version: str
    values_json: dict[str, float | None]
    masks_json: dict[str, bool]
    context_json: dict[str, Any]
    input_checksum: str | None
    created_at: datetime
    updated_at: datetime


class SequenceUpsert(StrictModel):
    schema_version: str = Field(default="sequence7.v4", min_length=1, max_length=64)
    tokens: list[SequenceToken] = Field(min_length=0, max_length=10000)
    event_uids: list[str] = Field(min_length=0, max_length=10000)
    side_fields: list[dict[str, Any]] = Field(default_factory=list, max_length=10000)
    input_checksum: str | None = Field(default=None, max_length=128)

    @field_validator("side_fields")
    @classmethod
    def computed_time_fields_are_server_owned(
        cls,
        side_fields: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        reserved = {"time_sin", "time_cos", "calendar_context"}
        for index, values in enumerate(side_fields):
            overlap = reserved & set(values)
            if overlap:
                raise ValueError(
                    f"side_fields[{index}] contains server-owned fields: {sorted(overlap)}"
                )
        return side_fields

    @model_validator(mode="after")
    def aligned_side_channels(self) -> SequenceUpsert:
        size = len(self.tokens)
        aligned = {
            "event_uids": len(self.event_uids),
        }
        if self.side_fields:
            aligned["side_fields"] = len(self.side_fields)
        if any(length != size for length in aligned.values()):
            raise ValueError(
                f"sequence side channels must align with tokens: {aligned}, tokens={size}"
            )
        return self


class SequenceRead(ORMModel):
    id: str
    user_id: str
    day: date
    schema_version: str
    tokens_json: list[str]
    pc_contexts_json: list[str]
    calendar_contexts_json: list[str]
    gap_buckets_json: list[str | None]
    event_uids_json: list[str]
    side_fields_json: list[dict[str, Any]]
    time_sin_json: list[float]
    time_cos_json: list[float]
    seq_len: int
    truncated: bool
    max_len: int
    input_checksum: str | None
    created_at: datetime
    updated_at: datetime


class ArtifactCreate(StrictModel):
    artifact_type: str = Field(
        pattern=r"^(00_events|01_feature128|02_sequences|03_role_context|04_branch_scores|"
        r"05_reference_tables|06_risk_alerts)$"
    )
    uri: str = Field(min_length=1, max_length=2048)
    schema_version: str = Field(min_length=1, max_length=128)
    checksum: str = Field(min_length=32, max_length=128)
    row_count: int = Field(ge=0)
    min_date: date | None = None
    max_date: date | None = None
    status: str = Field(default="READY", pattern=r"^(BUILDING|READY|FAILED|SUPERSEDED)$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingestion_job_id: str | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> ArtifactCreate:
        if self.min_date and self.max_date and self.max_date < self.min_date:
            raise ValueError("max_date must be on or after min_date")
        return self


class ArtifactRead(ORMModel):
    id: str
    artifact_type: str
    uri: str
    schema_version: str
    checksum: str
    row_count: int
    min_date: date | None
    max_date: date | None
    status: str
    metadata_json: dict[str, Any]
    ingestion_job_id: str | None
    created_at: datetime


class ReferenceProfileCreate(StrictModel):
    branch: Branch
    level: ReferenceLevel
    scope_key: str = Field(min_length=1, max_length=255)
    as_of_date: date
    model_version: str = Field(min_length=1, max_length=128)
    config_version: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    fitted_through: date
    frozen: bool = False
    support: dict[str, Any]
    statistics: dict[str, Any] = Field(default_factory=dict)
    checksum: str = Field(min_length=32, max_length=128)

    @model_validator(mode="after")
    def validate_scope(self) -> ReferenceProfileCreate:
        if self.level is ReferenceLevel.NO_SCORE:
            raise ValueError("NO_SCORE is not a storable reference level")
        expected_prefix = {
            ReferenceLevel.PERSON: "user:",
            ReferenceLevel.ROLE: "role:",
            ReferenceLevel.GLOBAL: "global",
        }[self.level]
        if not self.scope_key.startswith(expected_prefix):
            raise ValueError(
                f"{self.level} reference scope_key must start with {expected_prefix!r}"
            )
        if self.fitted_through >= self.as_of_date:
            raise ValueError("reference data must stop before as_of_date")
        return self


class ReferenceProfileRead(ORMModel):
    id: str
    branch: str
    level: str
    scope_key: str
    as_of_date: date
    model_version: str
    config_version: str
    version: str
    fitted_through: date
    frozen: bool
    support_json: dict[str, Any]
    statistics_json: dict[str, Any]
    checksum: str
    created_at: datetime


class BranchAssessmentInput(StrictModel):
    raw_score: float | None = None
    reference_profile_ids: dict[str, str] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    seq_len: int | None = Field(default=None, ge=0)
    truncated: bool | None = None

    @field_validator("raw_score")
    @classmethod
    def raw_score_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("raw_score must be finite")
        return value

class AssessmentCreate(StrictModel):
    user_id: str = Field(min_length=1, max_length=128)
    day: date
    model_version: str = Field(min_length=1, max_length=128)
    config_version: str = Field(default="framework.v4", min_length=1, max_length=128)
    split: str = Field(pattern=r"^(TRAIN|VALIDATION|TEST|PRODUCTION)$")
    feature: BranchAssessmentInput | None = None
    sequence: BranchAssessmentInput | None = None
    feature_weight: float = Field(default=0.5, ge=0, le=1)
    sequence_weight: float = Field(default=0.5, ge=0, le=1)
    alert_threshold: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_fusion_weights(self) -> AssessmentCreate:
        if self.feature_weight + self.sequence_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")
        return self


class BranchScoreRead(ORMModel):
    id: str
    branch: str
    raw_score: float | None
    calibrated_score: float | None
    selected_level: str
    reference_profile_id: str | None
    fallback_reasons_json: list[str]
    support_json: dict[str, Any]
    evidence_json: dict[str, Any]
    model_version: str
    config_version: str
    created_at: datetime


class AssessmentRead(ORMModel):
    id: str
    user_id: str
    day: date
    role_code: str | None
    split: str
    model_version: str
    config_version: str
    feature_score_id: str | None
    sequence_score_id: str | None
    feature_weight: float
    sequence_weight: float
    risk: float | None
    threshold: float | None
    state: str
    is_alert: bool
    no_score_reason: str | None
    created_at: datetime


class AssessmentDetail(BaseModel):
    assessment: AssessmentRead
    feature: BranchScoreRead | None
    sequence: BranchScoreRead | None
    alert: AlertRead | None = None
    safe_updates: list[SafeUpdateRead] = Field(default_factory=list)


class AlertRead(ORMModel):
    id: str
    assessment_id: str
    user_id: str
    day: date
    risk: float
    threshold: float
    severity: str
    status: str
    assignee: str | None
    resolution: str | None
    opened_at: datetime
    updated_at: datetime
    closed_at: datetime | None


class AlertPatch(StrictModel):
    status: str | None = Field(
        default=None, pattern=r"^(OPEN|ACKNOWLEDGED|INVESTIGATING|CLOSED|FALSE_POSITIVE)$"
    )
    assignee: str | None = Field(default=None, max_length=255)
    resolution: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def at_least_one_change(self) -> AlertPatch:
        if not self.model_fields_set:
            raise ValueError("at least one alert field must be supplied")
        if self.status in {"CLOSED", "FALSE_POSITIVE"} and not self.resolution:
            raise ValueError("resolution is required when closing an alert")
        return self


class SafeUpdateRead(ORMModel):
    id: str
    assessment_id: str
    user_id: str
    day: date
    role_code: str | None
    branch: str
    status: str
    quarantine_until: date
    decision_reasons_json: list[str]
    decided_at: datetime | None
    created_at: datetime


class SafeUpdateProcessRequest(StrictModel):
    through_date: date
    limit: int = Field(default=1000, ge=1, le=10000)


class SafeUpdateProcessResult(BaseModel):
    accepted: int
    rejected: int
    pending: int


class FrameworkInfo(BaseModel):
    feature_schema_version: str
    feature_dimension: int
    sequence_schema_version: str
    sequence_tokens: list[str]
    framework_config: dict[str, Any]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: Any | None = None
