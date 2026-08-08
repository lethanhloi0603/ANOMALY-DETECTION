"""Create the database schema, seed catalogs, and ensure the default organization."""

from __future__ import annotations

import logging

from app.database import init_db, session_scope
from app.services import ensure_default_organization


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("database-bootstrap")

    logger.info("Creating database schema and seeding locked catalogs")
    init_db()
    with session_scope() as session:
        organization = ensure_default_organization(session)
        organization_slug = organization.slug

    logger.info("Database bootstrap complete; organization=%s", organization_slug)


if __name__ == "__main__":
    main()
