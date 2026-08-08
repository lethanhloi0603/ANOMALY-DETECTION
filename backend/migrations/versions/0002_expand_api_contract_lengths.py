"""expand API contract string lengths

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _alter_release_lengths(table_name: str, *, upgrade: bool) -> None:
    old_length, new_length = ((120, 128) if upgrade else (128, 120))
    with op.batch_alter_table(table_name) as batch_op:
        for column_name in ("model_version", "config_version"):
            batch_op.alter_column(
                column_name,
                existing_type=sa.String(length=old_length),
                type_=sa.String(length=new_length),
                existing_nullable=False,
            )


def upgrade() -> None:
    with op.batch_alter_table("canonical_events") as batch_op:
        batch_op.alter_column(
            "original_id",
            existing_type=sa.String(length=240),
            type_=sa.String(length=255),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "pc",
            existing_type=sa.String(length=240),
            type_=sa.String(length=255),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "action",
            existing_type=sa.String(length=120),
            type_=sa.String(length=128),
            existing_nullable=False,
        )
    for table_name in ("reference_profiles", "branch_scores", "risk_assessments"):
        _alter_release_lengths(table_name, upgrade=True)
    with op.batch_alter_table("reference_profiles") as batch_op:
        batch_op.alter_column(
            "catalog_version",
            existing_type=sa.String(length=120),
            type_=sa.String(length=128),
            existing_nullable=False,
        )
    with op.batch_alter_table("alerts") as batch_op:
        batch_op.alter_column(
            "assignee",
            existing_type=sa.String(length=240),
            type_=sa.String(length=255),
            existing_nullable=True,
        )


def downgrade() -> None:
    connection = op.get_bind()
    checks = (
        ("canonical_events", "original_id", 240),
        ("canonical_events", "pc", 240),
        ("canonical_events", "action", 120),
        ("reference_profiles", "model_version", 120),
        ("reference_profiles", "config_version", 120),
        ("reference_profiles", "catalog_version", 120),
        ("branch_scores", "model_version", 120),
        ("branch_scores", "config_version", 120),
        ("risk_assessments", "model_version", 120),
        ("risk_assessments", "config_version", 120),
        ("alerts", "assignee", 240),
    )
    for table_name, column_name, maximum in checks:
        count = connection.scalar(
            sa.text(
                f"SELECT COUNT(*) FROM {table_name} "
                f"WHERE LENGTH({column_name}) > :maximum"
            ),
            {"maximum": maximum},
        )
        if count:
            raise RuntimeError(
                f"cannot downgrade: {table_name}.{column_name} contains values "
                f"longer than {maximum}"
            )
    with op.batch_alter_table("alerts") as batch_op:
        batch_op.alter_column(
            "assignee",
            existing_type=sa.String(length=255),
            type_=sa.String(length=240),
            existing_nullable=True,
        )
    with op.batch_alter_table("reference_profiles") as batch_op:
        batch_op.alter_column(
            "catalog_version",
            existing_type=sa.String(length=128),
            type_=sa.String(length=120),
            existing_nullable=False,
        )
    for table_name in ("risk_assessments", "branch_scores", "reference_profiles"):
        _alter_release_lengths(table_name, upgrade=False)
    with op.batch_alter_table("canonical_events") as batch_op:
        batch_op.alter_column(
            "action",
            existing_type=sa.String(length=128),
            type_=sa.String(length=120),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "pc",
            existing_type=sa.String(length=255),
            type_=sa.String(length=240),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "original_id",
            existing_type=sa.String(length=255),
            type_=sa.String(length=240),
            existing_nullable=False,
        )
