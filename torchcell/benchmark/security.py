# torchcell/benchmark/security.py
# [[torchcell.benchmark.security]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/security.py
# Test file: tests/torchcell/benchmark/test_security.py

"""Account security primitives: session tokens and the one-account-per-person rules.

- The service stores no passwords. A person proves who they are at CILogon
  (:mod:`torchcell.benchmark.oidc`); what is kept is the identity CILogon asserts.
- A session is a signed JWT (HS256) carrying only the user id, an issue time and an
  expiry. The signing secret comes from a file, never from the image. Other keys the
  service needs are derived from it per purpose (:func:`derive_key`), so no two uses
  share a key.
- A sign-in code is 32 random bytes, handed to the browser once and stored as its
  sha256. The account page trades it for a session token.
- One person, one account: an address is reduced to a canonical form before the unique
  check (lowercase, ``+tag`` removed, and dots removed for Gmail, which ignores them),
  so ``a.b+x@gmail.com`` and ``ab@gmail.com`` are the same account, whichever identity
  provider released them. An :class:`AccountPolicy` can also block listed email
  domains, require an allowed suffix, or accept only listed identity providers. This
  raises the cost of a second account; it cannot prove two identities belong to one
  person, which is what admin approval and the public per-user history are for.
- A client address is stored only as an HMAC, enough to count new accounts per address.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

import jwt
from pydantic import BaseModel, ConfigDict

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "tc-bench"
JWT_SECRET_MIN_BYTES = 32
DOT_INSENSITIVE_DOMAINS = frozenset({"gmail.com", "googlemail.com"})


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
    """Which email domains and identity providers may hold an account."""

    model_config = ConfigDict(frozen=True)

    blocked_domains: frozenset[str] = frozenset()
    allowed_domain_suffixes: tuple[str, ...] = ()
    allowed_idps: frozenset[str] = frozenset()

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

    def idp_allowed(self, idp: str | None) -> bool:
        """True when a sign-in through identity provider ``idp`` is accepted.

        With no list every provider is accepted. With a list, a sign-in that names no
        provider is refused, because it cannot be shown to come from a listed one.
        """
        return not self.allowed_idps or idp in self.allowed_idps


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


def derive_key(secret: str, purpose: str) -> str:
    """A key for ``purpose`` derived from ``secret`` (HMAC-SHA256, hex)."""
    return hmac.new(
        secret.encode("utf-8"), purpose.encode("utf-8"), "sha256"
    ).hexdigest()


def hash_client_address(address: str, secret: str) -> str:
    """A keyed hash of a client address, so new accounts can be counted without storing it."""
    return hmac.new(
        secret.encode("utf-8"), address.encode("utf-8"), "sha256"
    ).hexdigest()
