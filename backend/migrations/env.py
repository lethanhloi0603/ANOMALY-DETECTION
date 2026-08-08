"""Alembic environment for the operational core database."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app import models as _models  # noqa: F401
from app.database import Base
from app.settings import settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ConfigParser treats percent signs as interpolation markers.
config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without creating an Engine."""

    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=url.startswith("sqlite"),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live connection."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        if is_sqlite:
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            if connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() != 0:
                raise RuntimeError("could not disable SQLite foreign keys for batch migration")
        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                compare_server_default=True,
                render_as_batch=is_sqlite,
            )

            with context.begin_transaction():
                context.run_migrations()

            if is_sqlite:
                violations = connection.exec_driver_sql(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                if violations:
                    first = violations[0]
                    connection.rollback()
                    raise RuntimeError(
                        "SQLite foreign-key check failed after migration: "
                        f"table={first[0]!r}, rowid={first[1]!r}, parent={first[2]!r}"
                    )
                connection.commit()
        finally:
            if is_sqlite:
                if connection.in_transaction():
                    connection.rollback()
                connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                if connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() != 1:
                    raise RuntimeError("could not restore SQLite foreign keys after migration")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
