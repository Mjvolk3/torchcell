# tests/torchcell/benchmark/test_db.py
# [[tests.torchcell.benchmark.test_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_db.py
"""``torchcell.benchmark.db`` on an in-memory SQLite database.

What the schema must guarantee: three tables with the named indexes, a unique canonical
email (the one-account rule), timestamps that come back timezone-aware in UTC whatever
offset went in and that refuse a naive value, JSON columns that round-trip nested data,
and column defaults (unverified, unapproved, enabled, zero failed logins).
"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateTable

from torchcell.benchmark.db import (
    BOARD_STATUSES,
    Base,
    Submission,
    SubmissionStatus,
    User,
    init_schema,
    make_engine,
    make_session_factory,
    new_id,
    utcnow,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def sessions() -> sessionmaker[Session]:
    engine = make_engine("sqlite://")
    init_schema(engine)
    return make_session_factory(engine)


def _user(**overrides: Any) -> User:
    fields = {
        "email": "alice@example.org",
        "email_canonical": "alice@example.org",
        "display_name": "Alice",
        "password_hash": "hash",
        "created_at": NOW,
    }
    return User(**{**fields, **overrides})


def test_schema_tables_and_indexes() -> None:
    engine = make_engine("sqlite://")
    init_schema(engine)
    init_schema(engine)  # creating twice is a no-op
    inspector = inspect(engine)
    assert sorted(inspector.get_table_names()) == [
        "email_tokens",
        "submissions",
        "users",
    ]
    assert sorted(str(i["name"]) for i in inspector.get_indexes("submissions")) == [
        "ix_submissions_dataset_status",
        "ix_submissions_user_created",
    ]


def test_schema_renders_for_postgresql() -> None:
    # No server: the DDL is compiled for the PostgreSQL dialect and read as text.
    ddl = {
        table.name: str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for table in Base.metadata.sorted_tables
    }
    assert list(ddl) == ["users", "email_tokens", "submissions"]
    assert "val_scores JSONB" in ddl["submissions"]
    assert "flags JSONB NOT NULL" in ddl["submissions"]
    assert "created_at TIMESTAMP WITH TIME ZONE NOT NULL" in ddl["submissions"]
    assert "FOREIGN KEY(user_id) REFERENCES users (id)" in ddl["submissions"]
    assert "UNIQUE (email_canonical)" in ddl["users"]
    assert "ON DELETE CASCADE" in ddl["email_tokens"]


def test_status_values() -> None:
    assert [s.value for s in SubmissionStatus] == [
        "rejected",
        "provisional",
        "verified",
        "withdrawn",
    ]
    assert BOARD_STATUSES == (SubmissionStatus.PROVISIONAL, SubmissionStatus.VERIFIED)


def test_ids_and_clock() -> None:
    first, second = new_id(), new_id()
    assert len(first) == 32 and first != second
    assert int(first, 16) >= 0
    assert utcnow().tzinfo is UTC


def test_user_defaults(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_user())
        session.commit()
        user = session.scalars(select(User)).one()
        assert len(user.id) == 32
        assert (user.email_verified, user.approved, user.disabled, user.is_system) == (
            False,
            False,
            False,
            False,
        )
        assert user.failed_logins == 0
        assert user.locked_until is None
        assert user.affiliation is None


def test_canonical_email_is_unique(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_user())
        session.add(_user(email="a.lice@example.org"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_timestamps_round_trip_as_utc(sessions: sessionmaker[Session]) -> None:
    plus_five = timezone(timedelta(hours=5))
    with sessions() as session:
        session.add(_user(created_at=datetime(2026, 10, 1, 17, 0, 0, tzinfo=plus_five)))
        session.commit()
    with sessions() as session:
        created_at = session.scalars(select(User.created_at)).one()
        assert created_at == NOW
        assert created_at.utcoffset() == timedelta(0)


def test_naive_timestamp_is_refused(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_user(created_at=datetime(2026, 10, 1, 12, 0, 0)))
        with pytest.raises(StatementError, match="naive datetime"):
            session.commit()


def test_submission_json_columns_round_trip(sessions: sessionmaker[Session]) -> None:
    scores = {
        "n_records": 4,
        "macro": {"pearson": 0.5},
        "per_target": {"y": {"mse": 1.5}},
    }
    with sessions() as session:
        user = _user()
        session.add(user)
        session.flush()
        session.add(
            Submission(
                user_id=user.id,
                dataset_slug="toy-fitness",
                dataset_version="1",
                status=SubmissionStatus.PROVISIONAL,
                created_at=NOW,
                method_name="m",
                val_scores=scores,
                flags=["test_exceeds_val"],
            )
        )
        session.commit()
    with sessions() as session:
        stored = session.scalars(select(Submission)).one()
        assert stored.val_scores == scores
        assert stored.test_scores is None
        assert stored.flags == ["test_exceeds_val"]
        assert stored.rejection_reasons == []
        assert stored.is_baseline is False
        assert stored.status == "provisional"
        assert stored.user.display_name == "Alice"
        assert session.scalars(
            select(Submission).where(Submission.status.in_(BOARD_STATUSES))
        ).all() == [stored]
