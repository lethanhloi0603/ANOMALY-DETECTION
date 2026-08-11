"""classify legacy references and correct legacy candidate eligibility

Revision ID: 0004
Revises: 0003

Revision 0003 tagged pre-v5 candidates correctly, but copied
``quarantine_until`` directly into ``eligible_on``.  The persisted contract is
that the first eligible day is the day after the inclusive quarantine window.
Only legacy rows that still contain the equality produced by 0003 are repaired;
already-corrected rows and v5 candidates are left unchanged.

Pre-v5 ReferenceProfile rows also receive an explicit ``legacy`` release kind.
That value is profile provenance only and is forbidden for ReferenceRelease
materialization events.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_POLICY_VERSION = "framework.v4.safe_update.v1"
LEGACY_CONFIG_VERSION = "framework.v4"


def _reference_profiles() -> sa.Table:
    return sa.table(
        "reference_profiles",
        sa.column("config_version", sa.String(128)),
        sa.column("policy_version", sa.String(128)),
        sa.column("release_kind", sa.String(32)),
        sa.column("parent_reference_profile_id", sa.Uuid()),
        sa.column("calibration_parent_profile_id", sa.Uuid()),
        sa.column("reference_version", sa.Integer()),
        sa.column("release_day", sa.Date()),
        sa.column("release_influence_ratio", sa.Float()),
    )


def _true_legacy_profile(profiles: sa.Table) -> sa.ColumnElement[bool]:
    return sa.and_(
        profiles.c.config_version == LEGACY_CONFIG_VERSION,
        profiles.c.policy_version == LEGACY_CONFIG_VERSION,
        profiles.c.parent_reference_profile_id.is_(None),
        profiles.c.calibration_parent_profile_id.is_(None),
        profiles.c.reference_version.is_(None),
        profiles.c.release_day.is_(None),
        profiles.c.release_influence_ratio.is_(None),
    )


def upgrade() -> None:
    connection = op.get_bind()
    profiles = _reference_profiles()
    with op.batch_alter_table("reference_profiles") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_reference_profiles_reference_profile_release_kind"),
            type_="check",
        )
        batch_op.create_check_constraint(
            op.f("ck_reference_profiles_reference_profile_release_kind"),
            "release_kind IS NULL OR "
            "release_kind IN ('legacy', 'bootstrap', 'incremental')",
        )
    connection.execute(
        sa.update(profiles)
        .where(
            profiles.c.release_kind.is_(None),
            _true_legacy_profile(profiles),
        )
        .values(release_kind="legacy")
    )

    with op.batch_alter_table("reference_releases") as batch_op:
        batch_op.create_check_constraint(
            op.f("ck_reference_releases_reference_release_materialized_kind"),
            "kind IN ('bootstrap', 'incremental')",
        )

    candidates = sa.table(
        "safe_update_candidates",
        sa.column("policy_version", sa.String(128)),
        sa.column("eligible_on", sa.Date()),
        sa.column("quarantine_until", sa.Date()),
    )

    if connection.dialect.name == "postgresql":
        next_day = candidates.c.quarantine_until + 1
    elif connection.dialect.name == "sqlite":
        next_day = sa.func.date(candidates.c.quarantine_until, "+1 day")
    else:
        raise RuntimeError(
            "legacy candidate eligibility correction supports only PostgreSQL and SQLite"
        )

    connection.execute(
        sa.update(candidates)
        .where(
            candidates.c.policy_version == LEGACY_POLICY_VERSION,
            candidates.c.eligible_on == candidates.c.quarantine_until,
        )
        .values(eligible_on=next_day)
    )


def downgrade() -> None:
    connection = op.get_bind()
    profiles = _reference_profiles()
    invalid_legacy_profiles = int(
        connection.scalar(
            sa.select(sa.func.count())
            .select_from(profiles)
            .where(
                profiles.c.release_kind == "legacy",
                sa.not_(_true_legacy_profile(profiles)),
            )
        )
        or 0
    )
    if invalid_legacy_profiles:
        raise RuntimeError(
            "cannot downgrade: non-framework.v4 profiles use the legacy release kind"
        )

    connection.execute(
        sa.update(profiles)
        .where(
            profiles.c.release_kind == "legacy",
            _true_legacy_profile(profiles),
        )
        .values(release_kind=None)
    )
    with op.batch_alter_table("reference_profiles") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_reference_profiles_reference_profile_release_kind"),
            type_="check",
        )
        batch_op.create_check_constraint(
            op.f("ck_reference_profiles_reference_profile_release_kind"),
            "release_kind IS NULL OR release_kind IN ('bootstrap', 'incremental')",
        )
    with op.batch_alter_table("reference_releases") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_reference_releases_reference_release_materialized_kind"),
            type_="check",
        )

    # Reintroducing the old off-by-one eligibility value would corrupt evidence,
    # so the roll-forward data correction intentionally survives schema downgrade.
