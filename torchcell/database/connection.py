"""Client-side Neo4j connection settings, resolved from the environment."""

# torchcell/database/connection
# [[torchcell.database.connection]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/database/connection
# Test file: tests/torchcell/database/test_connection.py

import os

from pydantic import BaseModel

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USERNAME = "neo4j"
DEFAULT_PASSWORD = "torchcell"


class Neo4jConnectionSettings(BaseModel):
    """Where torchcell clients find the served knowledge graph.

    Resolved by :func:`neo4j_connection_settings` from ``NEO4J_URI``,
    ``NEO4J_USER``, and ``NEO4J_PASSWORD`` (the same names the serving
    container's ``database/.env`` uses), falling back to the local instance.
    One line in ``.env`` retargets every query path -- the serving host has
    moved twice (radiant -> GilaHyper) with the old host hardcoded at each
    call site, which is what this exists to end.
    """

    uri: str
    username: str
    password: str


def neo4j_connection_settings() -> Neo4jConnectionSettings:
    """Resolve connection settings from the environment at call time.

    Read at call time, not import time, so a caller's ``load_dotenv()`` (the
    established pattern for ``DATA_ROOT``) is honored regardless of import
    order.
    """
    return Neo4jConnectionSettings(
        uri=os.getenv("NEO4J_URI", DEFAULT_URI),
        username=os.getenv("NEO4J_USER", DEFAULT_USERNAME),
        password=os.getenv("NEO4J_PASSWORD", DEFAULT_PASSWORD),
    )
