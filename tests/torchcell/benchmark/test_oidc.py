# tests/torchcell/benchmark/test_oidc.py
# [[tests.torchcell.benchmark.test_oidc]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_oidc.py
"""``torchcell.benchmark.oidc``: the CILogon client's configuration and claim reading.

The protocol run itself (redirect, callback, token verification) is exercised through
the app in ``test_app.py`` against ``_fake_idp.py``. Here: the callback URL and what is
derived from it, the defaults that name CILogon, what an :class:`Identity` takes from
verified claims and when it refuses them, the display-name fallbacks, and that the
Authlib client is built with PKCE on and reaches the provider only through the
transport it is given.
"""

import asyncio
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from tests.torchcell.benchmark._fake_idp import (
    CLIENT_ID,
    CLIENT_SECRET,
    ISSUER,
    UIUC,
    UIUC_NAME,
    FakeIdp,
    person,
)
from torchcell.benchmark.oidc import (
    CILOGON_METADATA_URL,
    CILOGON_SCOPE,
    FALLBACK_DISPLAY_NAME,
    Identity,
    LoginError,
    LoginRefused,
    OidcConfig,
    build_provider,
    callback_url,
    identity_from_claims,
)


def _config(redirect_uri: str, **overrides: Any) -> OidcConfig:
    return OidcConfig(
        client_id=CLIENT_ID,
        client_secret=SecretStr(CLIENT_SECRET),
        redirect_uri=redirect_uri,
        **overrides,
    )


def _claims(**overrides: Any) -> dict[str, Any]:
    return {"iss": ISSUER, **person(), **overrides}


@pytest.mark.parametrize(
    ("public_url", "expected"),
    [
        ("https://bench.example", "https://bench.example/api/v1/auth/callback"),
        ("https://bench.example/", "https://bench.example/api/v1/auth/callback"),
        (
            "https://host.example/bench/",
            "https://host.example/bench/api/v1/auth/callback",
        ),
    ],
)
def test_callback_url(public_url: str, expected: str) -> None:
    assert callback_url(public_url, "/api/v1") == expected


def test_config_defaults_name_cilogon() -> None:
    config = _config("https://bench.example/api/v1/auth/callback")
    assert config.metadata_url == CILOGON_METADATA_URL
    assert config.metadata_url == "https://cilogon.org/.well-known/openid-configuration"
    assert config.scope == CILOGON_SCOPE
    assert config.scope.split() == [
        "openid",
        "email",
        "profile",
        "org.cilogon.userinfo",
    ]
    assert CLIENT_SECRET not in repr(config)
    with pytest.raises(ValidationError, match="frozen"):
        config.client_id = "another"


@pytest.mark.parametrize(
    ("redirect_uri", "secure", "cookie_path"),
    [
        ("https://bench.example/api/v1/auth/callback", True, "/api/v1/auth"),
        ("https://host.example/bench/api/v1/auth/callback", True, "/bench/api/v1/auth"),
        ("http://localhost:8725/api/v1/auth/callback", False, "/api/v1/auth"),
    ],
)
def test_cookie_scope_follows_the_callback(
    redirect_uri: str, secure: bool, cookie_path: str
) -> None:
    config = _config(redirect_uri)
    assert (config.secure, config.cookie_path) == (secure, cookie_path)


def test_identity_from_claims() -> None:
    identity = identity_from_claims(_claims(aud=CLIENT_ID, nonce="n", exp=1))
    assert identity == Identity(
        issuer=ISSUER,
        subject="http://cilogon.org/serverA/users/alice@example.org",
        email="alice@example.org",
        name="Alice",
        idp=UIUC,
        idp_name=UIUC_NAME,
    )


def test_identity_name_falls_back_to_given_and_family() -> None:
    claims = _claims(given_name="Ada", family_name="Lovelace")
    del claims["name"]
    assert identity_from_claims(claims).name == "Ada Lovelace"
    del claims["given_name"]
    assert identity_from_claims(claims).name == "Lovelace"
    del claims["family_name"]
    assert identity_from_claims(claims).name is None


def test_identity_without_provider_claims() -> None:
    claims = _claims(idp="", idp_name=None)
    identity = identity_from_claims(claims)
    assert (identity.idp, identity.idp_name) == (None, None)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"email": None}, LoginError.NO_EMAIL),
        ({"email": ""}, LoginError.NO_EMAIL),
        ({"email": "not-an-address"}, LoginError.FAILED),
        ({"sub": None}, LoginError.FAILED),
        ({"sub": "s" * 256}, LoginError.FAILED),
        ({"iss": None}, LoginError.FAILED),
    ],
)
def test_identity_refusals(overrides: dict[str, Any], error: LoginError) -> None:
    with pytest.raises(LoginRefused) as refused:
        identity_from_claims(_claims(**overrides))
    assert refused.value.error is error
    assert str(refused.value) == error.value


@pytest.mark.parametrize(
    ("name", "email", "display_name"),
    [
        ("Ada Lovelace", "ada@example.org", "Ada Lovelace"),
        ("  Ada  ", "ada@example.org", "Ada"),
        (None, "ada.lovelace@example.org", "ada.lovelace"),
        ("A", "ada@example.org", "ada"),  # one letter is shorter than a display name
        ("A", "a@example.org", FALLBACK_DISPLAY_NAME),
        ("N" * 80, "ada@example.org", "N" * 60),
    ],
)
def test_display_name(name: str | None, email: str, display_name: str) -> None:
    identity = Identity(issuer=ISSUER, subject="s", email=email, name=name)
    assert identity.display_name == display_name


def test_login_error_values_are_the_account_page_contract() -> None:
    assert [error.value for error in LoginError] == [
        "denied",
        "failed",
        "unavailable",
        "no_email",
        "idp_not_allowed",
        "email_not_allowed",
        "email_in_use",
        "too_many_accounts",
        "disabled",
    ]


def test_provider_uses_pkce_and_only_the_given_transport() -> None:
    idp = FakeIdp()
    provider = build_provider(
        _config(
            "https://bench.example/api/v1/auth/callback", metadata_url=idp.metadata_url
        ),
        idp.transport(),
    )
    assert provider.client_id == CLIENT_ID
    assert provider.client_kwargs["code_challenge_method"] == "S256"
    assert provider.client_kwargs["scope"] == CILOGON_SCOPE

    metadata = asyncio.run(provider.load_server_metadata())
    assert metadata["issuer"] == ISSUER
    assert metadata["token_endpoint"] == f"{ISSUER}/oauth2/token"
    keys = asyncio.run(provider.fetch_jwk_set())
    assert [key["kid"] for key in keys["keys"]] == ["idp-key-1"]
    assert "d" not in keys["keys"][0]  # the public half only
