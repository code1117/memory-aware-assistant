"""Provide the Neo4j connection foundation for the persistent Shared Brain.

Run `uv run python -m app.brain` to check connectivity without changing data.
"""

import argparse
from threading import Lock

from neo4j import Driver, GraphDatabase, Query
from neo4j.exceptions import DriverError, Neo4jError

from app.config import ConfigurationError, Neo4jSettings, get_neo4j_settings
from app.models import MemoryFact, UserProfile


class BrainError(Exception):
    """Describe a database failure without exposing credentials or raw errors."""


class BrainStore:
    """Save user profiles and facts using a shared, externally managed driver."""

    def __init__(self, driver: Driver, database: str) -> None:
        """Reuse the driver's connection pool; open short-lived sessions per query."""
        self._driver = driver
        self._database = database
        self._schema_ready = False
        self._schema_lock = Lock()

    def _query(self, cypher: str, **parameters) -> list[dict]:
        """Execute a parameterized query and consume results before closing its session."""
        try:
            with self._driver.session(database=self._database) as session:
                # User values are parameters, never concatenated into Cypher.
                # Consuming the result completes this auto-commit transaction.
                result = session.run(Query(cypher, timeout=10.0), parameters)
                return [record.data() for record in result]
        except (Neo4jError, DriverError) as exc:
            raise BrainError("The database operation failed. Check Neo4j connectivity and configuration.") from exc

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """Reject empty identifiers before reading or writing user-scoped data."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a nonblank string.")

    def ensure_schema(self) -> None:
        """Create uniqueness rules once at startup; repeated calls are safe."""
        with self._schema_lock:
            if self._schema_ready:
                return
            self._query("""
                CREATE CONSTRAINT user_id_unique IF NOT EXISTS
                FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """)
            self._query("""
                CREATE CONSTRAINT memory_user_key_unique IF NOT EXISTS
                FOR (m:Memory) REQUIRE (m.user_id, m.key) IS UNIQUE
            """)
            self._schema_ready = True

    def save_updates(
        self, user_id: str, profile: UserProfile, memories: list[MemoryFact]
    ) -> None:
        """Commit profile changes and all extracted facts together, or none on failure."""
        self._validate_user_id(user_id)
        properties = profile.model_dump(mode="json", exclude_unset=True)
        if not properties and not memories:
            return
        self._query("""
            MERGE (u:User {user_id: $user_id})
            ON CREATE SET u.created_at = datetime()
            SET u += $profile, u.updated_at = datetime()
            WITH u
            UNWIND $memories AS fact
            MERGE (m:Memory {user_id: $user_id, key: fact.key})
            ON CREATE SET m.created_at = datetime()
            SET m += fact, m.updated_at = datetime()
            MERGE (u)-[:HAS_MEMORY]->(m)
        """, user_id=user_id, profile=properties,
            memories=[memory.model_dump() for memory in memories])

    def save_profile(self, user_id: str, profile: UserProfile) -> None:
        """Create or patch a profile: omitted fields stay unchanged; explicit null clears a field."""
        self._validate_user_id(user_id)
        self._query("""
            MERGE (u:User {user_id: $user_id})
            ON CREATE SET u.created_at = datetime()
            SET u += $properties, u.updated_at = datetime()
        """, user_id=user_id, properties=profile.model_dump(mode="json", exclude_unset=True))

    def get_profile(self, user_id: str) -> UserProfile | None:
        """Read profile fields for one user; return None if that user does not exist."""
        self._validate_user_id(user_id)
        rows = self._query("""
            MATCH (u:User {user_id: $user_id})
            RETURN properties(u) AS profile
        """, user_id=user_id)
        if not rows:
            return None
        # Read existing properties rather than explicitly naming optional keys
        # Neo4j may never have seen. Exclude storage metadata before validation;
        # Pydantic supplies None for optional fields that have not been saved.
        properties = rows[0]["profile"]
        return UserProfile.model_validate({
            key: value for key, value in properties.items()
            if key in UserProfile.model_fields
        })

    def save_memory(self, user_id: str, memory: MemoryFact) -> None:
        """Insert a fact or replace the same user's fact with the same stable key."""
        self._validate_user_id(user_id)
        # MERGE finds or creates the node. SET updates its values. The user and
        # memory relationship are written together in a single transaction.
        self._query("""
            MERGE (u:User {user_id: $user_id})
            ON CREATE SET u.created_at = datetime()
            MERGE (m:Memory {user_id: $user_id, key: $key})
            ON CREATE SET m.created_at = datetime()
            SET m += $properties, m.updated_at = datetime()
            MERGE (u)-[:HAS_MEMORY]->(m)
        """, user_id=user_id, key=memory.key, properties=memory.model_dump())

    def get_memories(
        self, user_id: str, topics: list[str] | None = None, limit: int = 20
    ) -> list[MemoryFact]:
        """Read a bounded set of this user's facts, optionally filtered by topic."""
        self._validate_user_id(user_id)
        if not 1 <= limit <= 100:
            raise ValueError("Memory retrieval limit must be between 1 and 100.")
        rows = self._query("""
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            WHERE m.user_id = $user_id AND ($topics IS NULL OR properties(m)['topic'] IN $topics)
            RETURN properties(m) AS memory
            ORDER BY memory['updated_at'] DESC, memory['key']
            LIMIT $limit
        """, user_id=user_id, topics=topics, limit=limit)
        # Optional target_year/timeframe may be absent; exclude internal user
        # identifiers and timestamps while preserving the model's validation.
        return [MemoryFact.model_validate({
            key: value for key, value in row["memory"].items()
            if key in MemoryFact.model_fields
        }) for row in rows]


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
    """Check connectivity or run a small, explicitly requested persistent-storage demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="check",
        choices=["check", "save-demo", "read-demo", "update-demo"],
        help="Demo actions use only the reserved sample user storage-demo-rahul.",
    )
    action = parser.parse_args().action
    try:
        if action == "check":
            check_connection()
            print("Neo4j connection successful (connected=1).")
            return
        settings = get_neo4j_settings()
        with create_driver(settings) as driver:
            store = BrainStore(driver, settings.database)
            user_id = "storage-demo-rahul"
            if action in {"save-demo", "update-demo"}:
                store.ensure_schema()
                if action == "save-demo":
                    store.save_profile(user_id, UserProfile(name="Rahul", birth_place="Delhi"))
                year = 2027 if action == "save-demo" else 2028
                store.save_memory(user_id, MemoryFact(
                    key="career_change", kind="goal", topic="career",
                    content=f"Plans to change jobs in {year}.", target_year=year,
                ))
                print(f"Saved career_change with target year {year} for {user_id}.")
            else:
                profile = store.get_profile(user_id)
                memories = store.get_memories(user_id)
                print("Profile:", profile.model_dump_json(exclude_none=True) if profile else "not found")
                print(f"Memories: {len(memories)}")
                for memory in memories:
                    print(memory.model_dump_json(exclude_none=True))
    except (ConfigurationError, BrainError) as exc:
        raise SystemExit(f"Neo4j operation failed: {exc}") from None


if __name__ == "__main__":
    main()
