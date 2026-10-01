# torchcell/benchmark/security.py
# [[torchcell.benchmark.security]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/security.py
# Test file: tests/torchcell/benchmark/test_security.py

"""Account security primitives: passwords, tokens, and the one-account-per-person rules.

- Passwords are hashed with argon2id (``argon2-cffi`` defaults); the plaintext is never
  stored or logged.
- A session is a signed JWT (HS256) carrying only the user id, an issue time and an
  expiry. The signing secret comes from a file, never from the image.
- An email confirmation token is 32 random bytes, emailed once and stored as its sha256.
- One person, one account: an address is reduced to a canonical form before the unique
  check (lowercase, ``+tag`` removed, and dots removed for Gmail, which ignores them),
  so ``a.b+x@gmail.com`` and ``ab@gmail.com`` are the same account. An
  :class:`AccountPolicy` can also block listed domains or require an allowed suffix.
  This raises the cost of a second account; it cannot prove two addresses belong to
  one person, which is what admin approval and the public per-user history are for.
- A client address is stored only as an HMAC, enough to count signups per address.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from pydantic import BaseModel, ConfigDict

PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "tc-bench"
JWT_SECRET_MIN_BYTES = 32
DOT_INSENSITIVE_DOMAINS = frozenset({"gmail.com", "googlemail.com"})

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """The argon2id hash of ``password`` (salted; safe to store)."""
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """True when ``password`` matches ``password_hash``."""
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


# Verified against when the email is unknown, so a login attempt costs the same time
# whether or not the account exists.
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(16))


def canonical_email(email: str) -> str:
    """The form of ``email`` that the unique-account check compares."""
    local, _, domain = email.strip().lower().rpartition("@")
    local = local.split("+", 1)[0]
    if domain in DOT_INSENSITIVE_DOMAINS:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"


def email_domain(email: str) -> str:
    """The lowercase domain of ``email``."""
    return email.strip().lower().rpartition("@")[2]


class AccountPolicy(BaseModel):
    """Which email domains may register."""

    model_config = ConfigDict(frozen=True)

    blocked_domains: frozenset[str] = frozenset()
    allowed_domain_suffixes: tuple[str, ...] = ()

    def rejection_reason(self, email: str) -> str | None:
        """Why ``email`` may not register, or None when it may."""
        domain = email_domain(email)
        if domain in self.blocked_domains:
            return f"addresses at {domain} cannot register"
        if self.allowed_domain_suffixes and not domain.endswith(
            self.allowed_domain_suffixes
        ):
            allowed = ", ".join(self.allowed_domain_suffixes)
            return f"registration needs an address ending in one of: {allowed}"
        return None


def issue_access_token(
    user_id: str, secret: str, ttl: timedelta, now: datetime
) -> tuple[str, datetime]:
    """A signed session token for ``user_id`` and the time it expires."""
    expires_at = now + ttl
    token = jwt.encode(
        {"sub": user_id, "iat": now, "exp": expires_at, "iss": JWT_ISSUER},
        secret,
        algorithm=JWT_ALGORITHM,
    )
    return token, expires_at


def decode_access_token(token: str, secret: str) -> str | None:
    """The user id of a valid, unexpired token, else None."""
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"require": ["sub", "iat", "exp", "iss"]},
        )
    except jwt.InvalidTokenError:
        return None
    return str(claims["sub"])


def hash_token(token: str) -> str:
    """sha256 hex of a one-time token (what the database stores)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_one_time_token() -> tuple[str, str]:
    """A fresh one-time token and the hash that stores it."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


def hash_client_address(address: str, secret: str) -> str:
    """A keyed hash of a client address, so signups can be counted without storing it."""
    return hmac.new(
        secret.encode("utf-8"), address.encode("utf-8"), "sha256"
    ).hexdigest()
