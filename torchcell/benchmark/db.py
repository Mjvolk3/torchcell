# torchcell/benchmark/db.py
# [[torchcell.benchmark.db]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/db.py
# Test file: tests/torchcell/benchmark/test_db.py

"""The SQL schema of the benchmark service (SQLAlchemy 2, PostgreSQL in production).

Three tables:

- ``users``: one row per account. The account key is the identity CILogon asserts,
  ``(oidc_issuer, oidc_subject)``, which is unique. ``email_canonical`` is unique too
  and is what enforces one account per address family (see
  :func:`torchcell.benchmark.security.canonical_email`). No password is stored.
- ``login_codes``: one-time sign-in codes, stored as sha256, with an expiry. The
  callback issues one and the account page trades it for a session token.
- ``submissions``: one row per ATTEMPT, rejected ones included, because the quota counts
  attempts. A scored row holds its validation and test scores as JSON in the shape of
  :class:`torchcell.benchmark.grading.SplitScores`, its integrity flags, and the path
  and sha256 of its zip archive.

The same models run on SQLite in the tests. Two types hide the dialect difference:
:class:`UtcDateTime` returns timezone-aware UTC on both, and ``JsonColumn`` is JSONB on
PostgreSQL and JSON elsewhere. The schema is created with :func:`init_schema`
(``tc-bench-server --init-db``); there is no migration tool yet, so a column change
after the first deployment needs one (Alembic) before it ships.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.types import TypeDecorator

JsonColumn = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    """The current time, timezone-aware in UTC."""
    return datetime.now(UTC)


def new_id() -> str:
    """A random 32-character hex id."""
    return uuid.uuid4().hex


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp that is always timezone-aware UTC in Python.

    SQLite drops the offset on the way in and returns a naive value, which cannot be
    compared with an aware ``now``; this type refuses a naive value on write and
    restores UTC on read.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Require an aware value and store it in UTC."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; pass a timezone-aware value")
        return value.astimezone(UTC)

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Return the stored value as aware UTC."""
        if value is None:
            return None
        return (
            value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        )


class SubmissionStatus(StrEnum):
    """Life cycle of a submission."""

    REJECTED = "rejected"  # failed validation; counted by the quota, not on the board
    PROVISIONAL = "provisional"  # graded, not yet reproduced
    VERIFIED = "verified"  # reproduced from the submitter's code
    WITHDRAWN = "withdrawn"  # removed from the board by an admin


BOARD_STATUSES: tuple[SubmissionStatus, ...] = (
    SubmissionStatus.PROVISIONAL,
    SubmissionStatus.VERIFIED,
)


class Base(DeclarativeBase):
    """Declarative base of the benchmark schema."""


class User(Base):
    """An account, keyed by the identity its sign-in provider asserts."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("oidc_issuer", "oidc_subject", name="uq_users_oidc_identity"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # Both are null only for the system account that holds the baselines.
    oidc_issuer: Mapped[str | None] = mapped_column(String(255))
    oidc_subject: Mapped[str | None] = mapped_column(String(255))
    idp: Mapped[str | None] = mapped_column(String(255))
    idp_name: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(320))
    email_canonical: Mapped[str] = mapped_column(String(320), unique=True)
    display_name: Mapped[str] = mapped_column(String(60))
    affiliation: Mapped[str | None] = mapped_column(String(120))
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)
    signup_address_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UtcDateTime())

    submissions: Mapped[list[Submission]] = relationship(back_populates="user")


class LoginCode(Base):
    """A one-time sign-in code (the sha256, never the code)."""

    __tablename__ = "login_codes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    code_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime())
    used_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=utcnow)


class Submission(Base):
    """One submission attempt and, when it was scored, its result."""

    __tablename__ = "submissions"
    __table_args__ = (
        Index("ix_submissions_user_created", "user_id", "created_at"),
        Index("ix_submissions_dataset_status", "dataset_slug", "status"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    dataset_slug: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())

    method_name: Mapped[str] = mapped_column(String(120))
    model_family: Mapped[str | None] = mapped_column(String(120))
    encoding: Mapped[str | None] = mapped_column(String(120))
    code_url: Mapped[str | None] = mapped_column(Text)
    submission_metadata: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn)

    upload_sha256: Mapped[str | None] = mapped_column(String(64))
    n_rows: Mapped[int | None] = mapped_column(Integer)
    rejection_reasons: Mapped[list[str]] = mapped_column(JsonColumn, default=list)
    val_scores: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn)
    test_scores: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn)
    flags: Mapped[list[str]] = mapped_column(JsonColumn, default=list)
    archive_path: Mapped[str | None] = mapped_column(Text)
    archive_sha256: Mapped[str | None] = mapped_column(String(64))

    reviewed_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    reviewed_by: Mapped[str | None] = mapped_column(String(120))
    review_note: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="submissions")


def make_engine(database_url: str) -> Engine:
    """An engine for ``database_url`` with connection health checks on."""
    return create_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """A session factory bound to ``engine`` (objects stay readable after commit)."""
    return sessionmaker(engine, expire_on_commit=False)


def init_schema(engine: Engine) -> None:
    """Create every table that does not exist yet."""
    Base.metadata.create_all(engine)
