"""add safe-update v5 persistence foundation

Revision ID: 0003
Revises: 0002

The migration is deliberately additive. Existing candidates retain their v4
policy and eligibility semantics; evidence that was never captured is left
NULL rather than reconstructed.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_POLICY_VERSION = "framework.v4.safe_update.v1"


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(
        postgresql.JSONB(astext_type=sa.Text()),
        "postgresql",
    )


def _backfill_legacy_candidates() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE safe_update_candidates "
            "SET policy_version = 'framework.v4.safe_update.v1', "
            "model_version = ("
            "SELECT a.model_version FROM risk_assessments AS a "
            "WHERE a.id = safe_update_candidates.source_assessment_id"
            "), "
            "config_version = ("
            "SELECT a.config_version FROM risk_assessments AS a "
            "WHERE a.id = safe_update_candidates.source_assessment_id"
            "), "
            "eligible_on = quarantine_until, "
            "support_contribution_json = '{}'"
        )
    )


def upgrade() -> None:
    with op.batch_alter_table("reference_profiles") as batch_op:
        batch_op.add_column(sa.Column("parent_reference_profile_id", sa.Uuid()))
        batch_op.add_column(sa.Column("calibration_parent_profile_id", sa.Uuid()))
        batch_op.add_column(sa.Column("reference_version", sa.Integer()))
        batch_op.add_column(
            sa.Column(
                "release_kind",
                sa.Enum(
                    "bootstrap",
                    "incremental",
                    name="reference_profile_release_kind",
                    native_enum=False,
                    create_constraint=False,
                    length=32,
                ),
            )
        )
        batch_op.add_column(sa.Column("release_day", sa.Date()))
        batch_op.add_column(sa.Column("release_influence_ratio", sa.Float()))
        batch_op.add_column(sa.Column("policy_version", sa.String(128)))
        batch_op.create_foreign_key(
            "fk_reference_profiles_parent_reference_profile_id_reference_profiles",
            "reference_profiles",
            ["parent_reference_profile_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_reference_profiles_calibration_parent_profile_id_reference_profiles",
            "reference_profiles",
            ["calibration_parent_profile_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_check_constraint(
            op.f("ck_reference_profiles_reference_version_positive"),
            "reference_version IS NULL OR reference_version > 0",
        )
        batch_op.create_check_constraint(
            op.f("ck_reference_profiles_reference_release_influence_range"),
            "release_influence_ratio IS NULL OR "
            "(release_influence_ratio >= 0 AND release_influence_ratio <= 1)",
        )
    op.execute(
        sa.text(
            "UPDATE reference_profiles "
            "SET policy_version = config_version "
            "WHERE policy_version IS NULL"
        )
    )
    with op.batch_alter_table("reference_profiles") as batch_op:
        batch_op.alter_column(
            "policy_version",
            existing_type=sa.String(128),
            nullable=False,
        )
        batch_op.create_check_constraint(
            op.f("ck_reference_profiles_reference_profile_release_kind"),
            "release_kind IS NULL OR release_kind IN ('bootstrap', 'incremental')",
        )

    op.create_table(
        "personal_reference_accumulators",
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role_assignment_id", sa.Uuid(), nullable=False),
        sa.Column(
            "branch",
            sa.Enum(
                "feature",
                "sequence",
                name="personal_accumulator_branch",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("config_version", sa.String(128), nullable=False),
        sa.Column("catalog_version", sa.String(128), nullable=False),
        sa.Column("admission_reference_profile_id", sa.Uuid(), nullable=False),
        sa.Column("active_reference_profile_id", sa.Uuid()),
        sa.Column(
            "status",
            sa.Enum(
                "warming",
                "active",
                "closed",
                "compromised",
                name="personal_accumulator_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("release_sequence", sa.Integer(), nullable=False),
        sa.Column("last_release_day", sa.Date()),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "release_sequence >= 0",
            name=op.f(
                "ck_personal_reference_accumulators_"
                "personal_accumulator_release_sequence_nonnegative"
            ),
        ),
        sa.CheckConstraint(
            "status != 'active' OR active_reference_profile_id IS NOT NULL",
            name=op.f(
                "ck_personal_reference_accumulators_"
                "personal_accumulator_active_reference_required"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name=op.f(
                "fk_personal_reference_accumulators_organization_id_organizations"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_personal_reference_accumulators_user_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["role_assignment_id"],
            ["role_assignments.id"],
            name=op.f(
                "fk_personal_reference_accumulators_"
                "role_assignment_id_role_assignments"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["admission_reference_profile_id"],
            ["reference_profiles.id"],
            name=op.f(
                "fk_personal_reference_accumulators_"
                "admission_reference_profile_id_reference_profiles"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["active_reference_profile_id"],
            ["reference_profiles.id"],
            name=op.f(
                "fk_personal_reference_accumulators_"
                "active_reference_profile_id_reference_profiles"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_personal_reference_accumulators")),
        sa.UniqueConstraint(
            "organization_id",
            "user_id",
            "role_assignment_id",
            "branch",
            "model_version",
            "config_version",
            "catalog_version",
            name="personal_accumulator_scope_release",
        ),
    )
    with op.batch_alter_table("personal_reference_accumulators") as batch_op:
        batch_op.create_index(
            "ix_personal_accumulators_org_status",
            ["organization_id", "status"],
        )

    op.create_table(
        "scoring_watermarks",
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("config_version", sa.String(128), nullable=False),
        sa.Column("expected_assessments", sa.Integer(), nullable=False),
        sa.Column("persisted_assessments", sa.Integer(), nullable=False),
        sa.Column("universe_checksum", sa.String(128), nullable=False),
        sa.Column("assessment_set_checksum", sa.String(128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "expected_assessments >= 0",
            name=op.f(
                "ck_scoring_watermarks_scoring_watermark_expected_nonnegative"
            ),
        ),
        sa.CheckConstraint(
            "persisted_assessments = expected_assessments",
            name=op.f("ck_scoring_watermarks_scoring_watermark_complete"),
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name=op.f("fk_scoring_watermarks_organization_id_organizations"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_scoring_watermarks")),
        sa.UniqueConstraint(
            "organization_id",
            "day",
            "model_version",
            "config_version",
            name="scoring_watermark_release",
        ),
    )
    with op.batch_alter_table("scoring_watermarks") as batch_op:
        batch_op.create_index(
            "ix_scoring_watermarks_org_day",
            ["organization_id", "day"],
        )

    op.create_table(
        "reference_releases",
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("accumulator_id", sa.Uuid(), nullable=False),
        sa.Column("release_sequence", sa.Integer(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "bootstrap",
                "incremental",
                name="reference_release_kind",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("parent_reference_profile_id", sa.Uuid()),
        sa.Column("calibration_parent_profile_id", sa.Uuid(), nullable=False),
        sa.Column("child_reference_profile_id", sa.Uuid(), nullable=False),
        sa.Column("release_day", sa.Date(), nullable=False),
        sa.Column("parent_support", sa.Integer(), nullable=False),
        sa.Column("applied_candidate_count", sa.Integer(), nullable=False),
        sa.Column("influence_ratio", sa.Float(), nullable=False),
        sa.Column("rolling_anchor_support", sa.Integer(), nullable=False),
        sa.Column("rolling_applied_count_before", sa.Integer(), nullable=False),
        sa.Column("policy_version", sa.String(128), nullable=False),
        sa.Column("candidate_manifest_json", _json_type(), nullable=False),
        sa.Column("manifest_checksum", sa.String(128), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "release_sequence > 0",
            name=op.f("ck_reference_releases_reference_release_sequence_positive"),
        ),
        sa.CheckConstraint(
            "parent_support >= 0 AND applied_candidate_count > 0",
            name=op.f("ck_reference_releases_reference_release_support_positive"),
        ),
        sa.CheckConstraint(
            "influence_ratio >= 0 AND influence_ratio <= 1",
            name=op.f("ck_reference_releases_reference_release_influence_range"),
        ),
        sa.CheckConstraint(
            "rolling_anchor_support >= 0 AND rolling_applied_count_before >= 0",
            name=op.f("ck_reference_releases_reference_release_rolling_nonnegative"),
        ),
        sa.CheckConstraint(
            "(kind = 'bootstrap' AND parent_reference_profile_id IS NULL) OR "
            "(kind = 'incremental' AND parent_reference_profile_id IS NOT NULL)",
            name=op.f("ck_reference_releases_reference_release_parent_shape"),
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name=op.f("fk_reference_releases_organization_id_organizations"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["accumulator_id"],
            ["personal_reference_accumulators.id"],
            name=op.f(
                "fk_reference_releases_accumulator_id_"
                "personal_reference_accumulators"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["parent_reference_profile_id"],
            ["reference_profiles.id"],
            name=op.f(
                "fk_reference_releases_parent_reference_profile_id_reference_profiles"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["calibration_parent_profile_id"],
            ["reference_profiles.id"],
            name=op.f(
                "fk_reference_releases_"
                "calibration_parent_profile_id_reference_profiles"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["child_reference_profile_id"],
            ["reference_profiles.id"],
            name=op.f(
                "fk_reference_releases_child_reference_profile_id_reference_profiles"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reference_releases")),
        sa.UniqueConstraint(
            "accumulator_id",
            "release_sequence",
            name="reference_release_sequence",
        ),
        sa.UniqueConstraint(
            "child_reference_profile_id",
            name="reference_release_child_profile",
        ),
        sa.UniqueConstraint(
            "manifest_checksum",
            name="reference_release_manifest_checksum",
        ),
    )
    with op.batch_alter_table("reference_releases") as batch_op:
        batch_op.create_index(
            "ix_reference_releases_accumulator_day",
            ["accumulator_id", "release_day"],
        )

    with op.batch_alter_table("safe_update_candidates") as batch_op:
        batch_op.alter_column(
            "reference_profile_id",
            existing_type=sa.Uuid(),
            existing_nullable=False,
            nullable=True,
        )
        batch_op.add_column(sa.Column("accumulator_id", sa.Uuid()))
        batch_op.add_column(sa.Column("source_branch_score_id", sa.Uuid()))
        batch_op.add_column(sa.Column("source_feature_id", sa.Uuid()))
        batch_op.add_column(sa.Column("source_sequence_id", sa.Uuid()))
        batch_op.add_column(sa.Column("source_input_checksum", sa.String(128)))
        batch_op.add_column(sa.Column("admission_reference_profile_id", sa.Uuid()))
        batch_op.add_column(sa.Column("admission_reference_checksum", sa.String(128)))
        batch_op.add_column(sa.Column("admission_percentile", sa.Float()))
        batch_op.add_column(sa.Column("admission_threshold", sa.Float()))
        batch_op.add_column(sa.Column("eligible_on", sa.Date()))
        batch_op.add_column(sa.Column("support_contribution_json", _json_type()))
        batch_op.add_column(sa.Column("policy_version", sa.String(128)))
        batch_op.add_column(sa.Column("model_version", sa.String(128)))
        batch_op.add_column(sa.Column("config_version", sa.String(128)))
        batch_op.add_column(sa.Column("decision_at", sa.DateTime(timezone=True)))
        batch_op.add_column(sa.Column("materialized_reference_profile_id", sa.Uuid()))
        batch_op.add_column(sa.Column("reference_release_id", sa.Uuid()))
        batch_op.create_foreign_key(
            "fk_safe_update_candidates_accumulator_id_personal_reference_accumulators",
            "personal_reference_accumulators",
            ["accumulator_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_safe_update_candidates_source_branch_score_id_branch_scores",
            "branch_scores",
            ["source_branch_score_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_safe_update_candidates_source_feature_id_user_day_features",
            "user_day_features",
            ["source_feature_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_safe_update_candidates_source_sequence_id_user_day_sequences",
            "user_day_sequences",
            ["source_sequence_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_safe_update_candidates_admission_reference_profile_id_reference_profiles",
            "reference_profiles",
            ["admission_reference_profile_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_safe_update_candidates_"
            "materialized_reference_profile_id_reference_profiles",
            "reference_profiles",
            ["materialized_reference_profile_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_safe_update_candidates_reference_release_id_reference_releases",
            "reference_releases",
            ["reference_release_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_unique_constraint(
            "safe_update_day_per_accumulator",
            ["accumulator_id", "candidate_day"],
        )
        batch_op.create_unique_constraint(
            "safe_update_source_branch_score",
            ["source_branch_score_id"],
        )
        batch_op.create_check_constraint(
            op.f(
                "ck_safe_update_candidates_safe_update_admission_percentile_range"
            ),
            "admission_percentile IS NULL OR "
            "(admission_percentile >= 0 AND admission_percentile <= 1)",
        )
        batch_op.create_check_constraint(
            op.f(
                "ck_safe_update_candidates_safe_update_admission_threshold_range"
            ),
            "admission_threshold IS NULL OR "
            "(admission_threshold > 0 AND admission_threshold < 1)",
        )
        batch_op.create_check_constraint(
            op.f(
                "ck_safe_update_candidates_safe_update_eligible_after_quarantine"
            ),
            "eligible_on IS NULL OR eligible_on >= quarantine_until",
        )

    _backfill_legacy_candidates()
    with op.batch_alter_table("safe_update_candidates") as batch_op:
        batch_op.alter_column(
            "support_contribution_json",
            existing_type=_json_type(),
            nullable=False,
        )
        batch_op.alter_column(
            "policy_version",
            existing_type=sa.String(128),
            nullable=False,
        )
        batch_op.create_index(
            "ix_safe_update_candidates_policy_eligible",
            ["policy_version", "status", "eligible_on"],
        )


def _require_safe_downgrade() -> None:
    connection = op.get_bind()
    protected_tables = (
        "reference_releases",
        "personal_reference_accumulators",
        "scoring_watermarks",
    )
    for table_name in protected_tables:
        if connection.scalar(sa.text(f"SELECT COUNT(*) FROM {table_name}")):
            raise RuntimeError(
                f"cannot downgrade: {table_name} contains safe-update v5 state"
            )
    v5_candidates = connection.scalar(
        sa.text(
            "SELECT COUNT(*) FROM safe_update_candidates "
            "WHERE policy_version != :legacy_policy OR reference_profile_id IS NULL"
        ),
        {"legacy_policy": LEGACY_POLICY_VERSION},
    )
    if v5_candidates:
        raise RuntimeError("cannot downgrade: v5 or cold-start candidates exist")
    released_profiles = connection.scalar(
        sa.text(
            "SELECT COUNT(*) FROM reference_profiles WHERE release_kind IS NOT NULL"
        )
    )
    if released_profiles:
        raise RuntimeError("cannot downgrade: versioned Personal references exist")


def downgrade() -> None:
    _require_safe_downgrade()

    with op.batch_alter_table("safe_update_candidates") as batch_op:
        batch_op.drop_index("ix_safe_update_candidates_policy_eligible")
        batch_op.drop_constraint(
            op.f(
                "ck_safe_update_candidates_safe_update_eligible_after_quarantine"
            ),
            type_="check",
        )
        batch_op.drop_constraint(
            op.f(
                "ck_safe_update_candidates_safe_update_admission_threshold_range"
            ),
            type_="check",
        )
        batch_op.drop_constraint(
            op.f(
                "ck_safe_update_candidates_safe_update_admission_percentile_range"
            ),
            type_="check",
        )
        batch_op.drop_constraint("safe_update_source_branch_score", type_="unique")
        batch_op.drop_constraint("safe_update_day_per_accumulator", type_="unique")
        batch_op.drop_constraint(
            "fk_safe_update_candidates_reference_release_id_reference_releases",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_safe_update_candidates_"
            "materialized_reference_profile_id_reference_profiles",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_safe_update_candidates_admission_reference_profile_id_reference_profiles",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_safe_update_candidates_source_sequence_id_user_day_sequences",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_safe_update_candidates_source_feature_id_user_day_features",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_safe_update_candidates_source_branch_score_id_branch_scores",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_safe_update_candidates_accumulator_id_personal_reference_accumulators",
            type_="foreignkey",
        )
        for column_name in (
            "reference_release_id",
            "materialized_reference_profile_id",
            "decision_at",
            "config_version",
            "model_version",
            "policy_version",
            "support_contribution_json",
            "eligible_on",
            "admission_threshold",
            "admission_percentile",
            "admission_reference_checksum",
            "admission_reference_profile_id",
            "source_input_checksum",
            "source_sequence_id",
            "source_feature_id",
            "source_branch_score_id",
            "accumulator_id",
        ):
            batch_op.drop_column(column_name)
        batch_op.alter_column(
            "reference_profile_id",
            existing_type=sa.Uuid(),
            existing_nullable=True,
            nullable=False,
        )

    with op.batch_alter_table("reference_releases") as batch_op:
        batch_op.drop_index("ix_reference_releases_accumulator_day")
    op.drop_table("reference_releases")

    with op.batch_alter_table("scoring_watermarks") as batch_op:
        batch_op.drop_index("ix_scoring_watermarks_org_day")
    op.drop_table("scoring_watermarks")

    with op.batch_alter_table("personal_reference_accumulators") as batch_op:
        batch_op.drop_index("ix_personal_accumulators_org_status")
    op.drop_table("personal_reference_accumulators")

    with op.batch_alter_table("reference_profiles") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_reference_profiles_reference_profile_release_kind"),
            type_="check",
        )
        batch_op.drop_constraint(
            op.f("ck_reference_profiles_reference_release_influence_range"),
            type_="check",
        )
        batch_op.drop_constraint(
            op.f("ck_reference_profiles_reference_version_positive"),
            type_="check",
        )
        batch_op.drop_constraint(
            "fk_reference_profiles_calibration_parent_profile_id_reference_profiles",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_reference_profiles_parent_reference_profile_id_reference_profiles",
            type_="foreignkey",
        )
        for column_name in (
            "policy_version",
            "release_influence_ratio",
            "release_day",
            "release_kind",
            "reference_version",
            "calibration_parent_profile_id",
            "parent_reference_profile_id",
        ):
            batch_op.drop_column(column_name)
