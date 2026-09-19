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
# The knowledge-graph VERSION a client wants: ``latest`` (an alias the served DBMS
# retargets at each publish), ``pinned``, a release id such as ``2026.09.17-7715ee35``,
# or a ``major.minor`` version. Resolved to a database name by
# ``torchcell.knowledge_graphs.releases.resolve_database``.
DEFAULT_VERSION = "latest"


class Neo4jConnectionSettings(BaseModel):
    """Where torchcell clients find the served knowledge graph, and which version.

    Resolved by :func:`neo4j_connection_settings` from ``NEO4J_URI``,
    ``NEO4J_USER``, ``NEO4J_PASSWORD`` (the same names the serving
    container's ``database/.env`` uses) and ``TORCHCELL_KG_VERSION``, falling
    back to the local instance and ``latest``. One line in ``.env`` retargets
    every query path -- the serving host has moved twice (radiant -> GilaHyper)
    with the old host hardcoded at each call site, which is what this exists to
    end -- and one more pins a whole run to a release.
    """

    uri: str
    username: str
    password: str
    version: str = DEFAULT_VERSION


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
        version=os.getenv("TORCHCELL_KG_VERSION", DEFAULT_VERSION),
    )
