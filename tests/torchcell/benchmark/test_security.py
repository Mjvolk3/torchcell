# tests/torchcell/benchmark/test_security.py
# [[tests.torchcell.benchmark.test_security]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_security.py
"""``torchcell.benchmark.security``: tokens, derived keys, and the account policy.

The canonical form is what the unique-account check compares, so the cases that matter
are the ones that would otherwise let one person hold two accounts: a ``+tag``, Gmail
dots, ``googlemail.com``, and letter case. A token signed with another secret, a token
past its expiry, and a token from another issuer all decode to None. The module holds
no password code: sign-in is CILogon's, tested in ``test_oidc.py`` and ``test_app.py``.
"""

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from torchcell.benchmark.security import (
    JWT_ALGORITHM,
    JWT_ISSUER,
    AccountPolicy,
    canonical_email,
    decode_access_token,
    derive_key,
    email_domain,
    hash_client_address,
    hash_token,
    is_api_token,
    issue_access_token,
    new_api_token,
    new_one_time_token,
)

SECRET = "s" * 40
NOW = datetime.now(UTC).replace(microsecond=0)


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


def test_account_policy_identity_providers() -> None:
    uiuc = "urn:mace:incommon:uiuc.edu"
    anyone = AccountPolicy()
    assert anyone.idp_allowed(uiuc) is True
    assert anyone.idp_allowed(None) is True
    listed = AccountPolicy(allowed_idps=frozenset({uiuc}))
    assert listed.idp_allowed(uiuc) is True
    assert listed.idp_allowed("http://google.com/accounts/o8/id") is False
    # With a list, a sign-in that names no provider cannot be shown to be listed.
    assert listed.idp_allowed(None) is False


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

    # An unsigned token (``alg: none``), assembled by hand: header, claims, no signature.
    def segment(data: dict[str, object]) -> str:
        raw = json.dumps(data).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    stamp = int(NOW.timestamp())
    unsigned = ".".join(
        [
            segment({"alg": "none", "typ": "JWT"}),
            segment(
                {"sub": "user-1", "iat": stamp, "exp": stamp + 3600, "iss": JWT_ISSUER}
            ),
            "",
        ]
    )
    assert jwt.decode(unsigned, options={"verify_signature": False})["sub"] == "user-1"
    assert decode_access_token(unsigned, SECRET) is None


def test_one_time_token_is_stored_as_its_hash() -> None:
    token, stored = new_one_time_token()
    assert stored == hashlib.sha256(token.encode()).hexdigest() == hash_token(token)
    assert len(token) == 43  # 32 random bytes, urlsafe base64 without padding
    assert new_one_time_token()[0] != token


def test_derived_keys_differ_by_purpose_and_secret() -> None:
    key = derive_key(SECRET, "login-state")
    assert key == hmac.new(SECRET.encode(), b"login-state", "sha256").hexdigest()
    assert len(key) == 64
    assert key != SECRET
    assert key != derive_key(SECRET, "another-purpose")
    assert key != derive_key("t" * 40, "login-state")


def test_client_address_hash_is_keyed() -> None:
    hashed = hash_client_address("203.0.113.7", SECRET)
    assert hashed == hmac.new(SECRET.encode(), b"203.0.113.7", "sha256").hexdigest()
    assert hashed != hash_client_address("203.0.113.7", "t" * 40)
    assert hashed != hash_client_address("203.0.113.8", SECRET)


def test_api_token_is_prefixed_hashed_and_distinct_from_a_session_token() -> None:
    token, token_hash, hint = new_api_token()
    assert token.startswith("tcb_") and len(token) == 4 + 43  # 32 bytes, base64url
    assert token_hash == hashlib.sha256(token.encode()).hexdigest()
    assert hint == token[:12]
    assert new_api_token()[0] != token
    assert is_api_token(token)
    session_token, _ = issue_access_token("u1", SECRET, timedelta(hours=1), NOW)
    assert not is_api_token(session_token)
