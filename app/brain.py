"""Provide the Neo4j connection foundation for the persistent Shared Brain.

Run `uv run python -m app.brain` to check connectivity without changing data.
"""

from neo4j import Driver, GraphDatabase, Query
from neo4j.exceptions import DriverError, Neo4jError

from app.config import ConfigurationError, Neo4jSettings, get_neo4j_settings


class BrainError(Exception):
    """Describe a database failure without exposing credentials or raw errors."""


def create_driver(settings: Neo4jSettings) -> Driver:
    """Create a reusable driver; its caller must close it when finished."""
    try:
        # Driver creation configures a connection pool. Actual connectivity is
        # checked separately; construction alone does not prove the DB is ready.
        return GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
            connection_timeout=5.0,
            connection_acquisition_timeout=10.0,
            max_transaction_retry_time=0.0,
        )
    except (Neo4jError, DriverError, ValueError) as exc:
        raise BrainError("Could not configure the Neo4j driver. Check your database settings.") from exc


def check_connection() -> None:
    """Verify authentication and a read-only query against the configured database."""
    settings = get_neo4j_settings()
    try:
        # Context managers close the session and driver even if a check fails.
        with create_driver(settings) as driver:
            driver.verify_connectivity()
            # Verify the configured database too, not just the server address.
            with driver.session(database=settings.database) as session:
                record = session.run(Query("RETURN 1 AS connected", timeout=5.0)).single()
                if record is None or record["connected"] != 1:
                    raise BrainError("Neo4j returned an unexpected connection-check result.")
    except (Neo4jError, DriverError) as exc:
        raise BrainError(
            "Could not connect to Neo4j. Check that Docker and the database are "
            "running, and verify the address, credentials, and database name in .env."
        ) from exc


def main() -> None:
    """Print a safe check result and exit unsuccessfully if configuration or connection fails."""
    try:
        check_connection()
    except (ConfigurationError, BrainError) as exc:
        raise SystemExit(f"Neo4j connection check failed: {exc}") from None
    print("Neo4j connection successful (connected=1).")


if __name__ == "__main__":
    main()
