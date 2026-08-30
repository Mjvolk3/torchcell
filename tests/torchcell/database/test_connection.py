"""Tests for the environment-resolved Neo4j connection settings."""

# tests/torchcell/database/test_connection
# [[tests.torchcell.database.test_connection]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_connection

import pytest

from torchcell.database.connection import (
    DEFAULT_PASSWORD,
    DEFAULT_URI,
    DEFAULT_USERNAME,
    neo4j_connection_settings,
)


def test_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no NEO4J_* vars set, settings fall back to the local instance."""
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    settings = neo4j_connection_settings()
    assert settings.uri == DEFAULT_URI
    assert settings.username == DEFAULT_USERNAME
    assert settings.password == DEFAULT_PASSWORD


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEO4J_* vars override every field."""
    monkeypatch.setenv("NEO4J_URI", "bolt://gilahyper.zapto.org:7687")
    monkeypatch.setenv("NEO4J_USER", "reader")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    settings = neo4j_connection_settings()
    assert settings.uri == "bolt://gilahyper.zapto.org:7687"
    assert settings.username == "reader"
    assert settings.password == "secret"


def test_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later env change is reflected on the next call (no import-time freeze)."""
    monkeypatch.setenv("NEO4J_URI", "bolt://first:7687")
    assert neo4j_connection_settings().uri == "bolt://first:7687"
    monkeypatch.setenv("NEO4J_URI", "bolt://second:7687")
    assert neo4j_connection_settings().uri == "bolt://second:7687"
