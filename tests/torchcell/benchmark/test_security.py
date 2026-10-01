# tests/torchcell/benchmark/test_security.py
# [[tests.torchcell.benchmark.test_security]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_security.py
"""``torchcell.benchmark.security``: hashing, tokens, and address canonicalization.

The canonical form is what the unique-account check compares, so the cases that matter
are the ones that would otherwise let one person hold two accounts: a ``+tag``, Gmail
dots, ``googlemail.com``, and letter case. A token signed with another secret, a token
past its expiry, and a token from another issuer all decode to None.
"""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from torchcell.benchmark.security import (
    DUMMY_PASSWORD_HASH,
    JWT_ALGORITHM,
    JWT_ISSUER,
    AccountPolicy,
    canonical_email,
    decode_access_token,
    email_domain,
    hash_client_address,
    hash_password,
    hash_token,
    issue_access_token,
    new_one_time_token,
    verify_password,
)

SECRET = "s" * 40
NOW = datetime.now(UTC).replace(microsecond=0)


def test_password_hash_round_trip() -> None:
    stored = hash_password("correct horse battery")
    assert stored.startswith("$argon2id$")
    assert stored != hash_password("correct horse battery")  # salted
    assert verify_password(stored, "correct horse battery") is True
    assert verify_password(stored, "correct horse batterY") is False
    assert verify_password("not-a-hash", "anything") is False
    assert verify_password(DUMMY_PASSWORD_HASH, "anything") is False


@pytest.mark.parametrize(
    ("email", "canonical"),
    [
        ("Alice@Example.org", "alice@example.org"),
        ("  alice+bench@example.org ", "alice@example.org"),
        ("a.l.i.c.e@gmail.com", "alice@gmail.com"),
        ("a.lice+x@googlemail.com", "alice@gmail.com"),
        ("a.lice@example.org", "a.lice@example.org"),  # dots matter outside Gmail
    ],
)
def test_canonical_email(email: str, canonical: str) -> None:
    assert canonical_email(email) == canonical


def test_email_domain() -> None:
    assert email_domain("Bob@Lab.Example.EDU") == "lab.example.edu"


def test_account_policy() -> None:
    assert AccountPolicy().rejection_reason("x@anything.io") is None
    blocked = AccountPolicy(blocked_domains=frozenset({"mailinator.com"}))
    assert (
        blocked.rejection_reason("x@Mailinator.com")
        == "addresses at mailinator.com cannot register"
    )
    assert blocked.rejection_reason("x@example.org") is None
    academic = AccountPolicy(allowed_domain_suffixes=(".edu", ".ac.uk"))
    assert academic.rejection_reason("x@illinois.edu") is None
    assert (
        academic.rejection_reason("x@gmail.com")
        == "registration needs an address ending in one of: .edu, .ac.uk"
    )


def test_access_token_round_trip() -> None:
    token, expires_at = issue_access_token("user-1", SECRET, timedelta(hours=12), NOW)
    assert expires_at == NOW + timedelta(hours=12)
    assert decode_access_token(token, SECRET) == "user-1"
    claims = jwt.decode(token, SECRET, algorithms=[JWT_ALGORITHM], issuer=JWT_ISSUER)
    assert set(claims) == {"sub", "iat", "exp", "iss"}


def test_access_token_rejections() -> None:
    token, _ = issue_access_token("user-1", SECRET, timedelta(hours=12), NOW)
    assert decode_access_token(token, "t" * 40) is None
    assert decode_access_token(token + "x", SECRET) is None
    assert decode_access_token("not.a.token", SECRET) is None
    expired, _ = issue_access_token(
        "user-1", SECRET, timedelta(hours=1), NOW - timedelta(hours=2)
    )
    assert decode_access_token(expired, SECRET) is None
    foreign = jwt.encode(
        {"sub": "user-1", "iat": NOW, "exp": NOW + timedelta(hours=1), "iss": "other"},
        SECRET,
        algorithm=JWT_ALGORITHM,
    )
    assert decode_access_token(foreign, SECRET) is None
    no_expiry = jwt.encode(
        {"sub": "user-1", "iat": NOW, "iss": JWT_ISSUER},
        SECRET,
        algorithm=JWT_ALGORITHM,
    )
    assert decode_access_token(no_expiry, SECRET) is None
    unsigned = jwt.encode(
        {
            "sub": "user-1",
            "iat": NOW,
            "exp": NOW + timedelta(hours=1),
            "iss": JWT_ISSUER,
        },
        None,
        algorithm="none",
    )
    assert decode_access_token(unsigned, SECRET) is None


def test_one_time_token_is_stored_as_its_hash() -> None:
    token, stored = new_one_time_token()
    assert stored == hashlib.sha256(token.encode()).hexdigest() == hash_token(token)
    assert len(token) == 43  # 32 random bytes, urlsafe base64 without padding
    assert new_one_time_token()[0] != token


def test_client_address_hash_is_keyed() -> None:
    hashed = hash_client_address("203.0.113.7", SECRET)
    assert hashed == hmac.new(SECRET.encode(), b"203.0.113.7", "sha256").hexdigest()
    assert hashed != hash_client_address("203.0.113.7", "t" * 40)
    assert hashed != hash_client_address("203.0.113.8", SECRET)
