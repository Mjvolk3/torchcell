# tests/torchcell/benchmark/test_db.py
# [[tests.torchcell.benchmark.test_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_db.py
"""``torchcell.benchmark.db`` on an in-memory SQLite database.

What the schema must guarantee: three tables with the named indexes, a unique sign-in
identity and a unique canonical email (the one-account rules), no password column,
one-time sign-in codes that are unique and go with their account, timestamps that come
back timezone-aware in UTC whatever offset went in and that refuse a naive value, JSON
columns that round-trip nested data, and column defaults (unapproved, enabled).
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
    LoginCode,
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
        "oidc_issuer": "https://cilogon.org",
        "oidc_subject": "http://cilogon.org/serverA/users/1",
        "created_at": NOW,
    }
    return User(**{**fields, **overrides})


def test_schema_tables_and_indexes() -> None:
    engine = make_engine("sqlite://")
    init_schema(engine)
    init_schema(engine)  # creating twice is a no-op
    inspector = inspect(engine)
    assert sorted(inspector.get_table_names()) == [
        "api_tokens",
        "login_codes",
        "submissions",
        "users",
    ]
    columns = {str(column["name"]) for column in inspector.get_columns("users")}
    assert {"oidc_issuer", "oidc_subject", "idp", "idp_name"} <= columns
    assert not any("password" in name for name in columns)
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
    assert set(ddl) == {"users", "api_tokens", "login_codes", "submissions"}
    assert "UNIQUE (token_sha256)" in ddl["api_tokens"]
    assert "ON DELETE CASCADE" in ddl["api_tokens"]
    assert "val_scores JSONB" in ddl["submissions"]
    assert "flags JSONB NOT NULL" in ddl["submissions"]
    assert "created_at TIMESTAMP WITH TIME ZONE NOT NULL" in ddl["submissions"]
    assert "FOREIGN KEY(user_id) REFERENCES users (id)" in ddl["submissions"]
    assert "UNIQUE (email_canonical)" in ddl["users"]
    assert (
        "CONSTRAINT uq_users_oidc_identity UNIQUE (oidc_issuer, oidc_subject)"
        in ddl["users"]
    )
    assert "ON DELETE CASCADE" in ddl["login_codes"]
    assert "UNIQUE (code_sha256)" in ddl["login_codes"]


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
        assert (user.approved, user.disabled, user.is_system) == (False, False, False)
        assert user.affiliation is None
        assert user.idp is None and user.idp_name is None
        assert user.last_login_at is None


def test_canonical_email_is_unique(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_user())
        session.add(_user(email="a.lice@example.org", oidc_subject="another"))
        with pytest.raises(IntegrityError, match="email_canonical"):
            session.commit()


def test_sign_in_identity_is_unique(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_user())
        session.add(_user(email_canonical="bob@example.org"))
        with pytest.raises(IntegrityError, match="oidc_issuer"):
            session.commit()


def test_accounts_without_an_identity_can_coexist(
    sessions: sessionmaker[Session],
) -> None:
    # The baselines account has no sign-in identity; null pairs do not collide.
    with sessions() as session:
        for name in ("one", "two"):
            session.add(
                _user(
                    email_canonical=f"{name}@torchcell.invalid",
                    oidc_issuer=None,
                    oidc_subject=None,
                )
            )
        session.commit()
        assert session.scalars(select(User.oidc_subject)).all() == [None, None]


def test_login_codes_are_unique_and_belong_to_an_account(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as session:
        user = _user()
        session.add(user)
        session.flush()
        session.add(
            LoginCode(
                user_id=user.id,
                code_sha256="a" * 64,
                expires_at=NOW + timedelta(minutes=2),
                created_at=NOW,
            )
        )
        session.commit()
        stored = session.scalars(select(LoginCode)).one()
        assert stored.user_id == user.id
        assert stored.used_at is None
        assert stored.expires_at - stored.created_at == timedelta(minutes=2)
        session.add(
            LoginCode(
                user_id=user.id,
                code_sha256="a" * 64,
                expires_at=NOW + timedelta(minutes=2),
                created_at=NOW,
            )
        )
        with pytest.raises(IntegrityError, match="code_sha256"):
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
