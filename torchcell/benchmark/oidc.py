# torchcell/benchmark/oidc.py
# [[torchcell.benchmark.oidc]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/oidc.py
# Test file: tests/torchcell/benchmark/test_oidc.py

"""Sign-in through CILogon (OpenID Connect), with Authlib running the protocol.

The service keeps no passwords. A person signs in at CILogon with their institution
(or with ORCID, GitHub, Google or Microsoft), and CILogon tells the service who they
are in a signed ID token. Authlib's Starlette client does every protocol step: it
reads CILogon's discovery document, builds the authorization request with a ``state``,
a ``nonce`` and a PKCE challenge (S256), exchanges the code at the token endpoint, and
verifies the ID token's signature against CILogon's published keys along with its
issuer, audience, expiry and nonce. This module only configures that client and turns
verified claims into an :class:`Identity`.

What the service reads from the claims:

- ``iss`` and ``sub`` name the person. CILogon's ``sub`` is stable for one person at
  one identity provider, so the pair is the account key.
- ``email`` is required; a provider that does not release one is refused with a reason.
- ``idp`` (the provider's entity id) and ``idp_name`` record where the person signed
  in. An :class:`~torchcell.benchmark.security.AccountPolicy` can restrict ``idp``.
- ``name``, or ``given_name`` and ``family_name``, seed the display name.

A refused sign-in carries a :class:`LoginError`; the account page maps each value to a
sentence, so the API never puts free text from a provider into a redirect.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any
from urllib.parse import urlsplit

from authlib.integrations.starlette_client import OAuth, StarletteOAuth2App
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    SecretStr,
    StringConstraints,
    ValidationError,
)

CILOGON_METADATA_URL = "https://cilogon.org/.well-known/openid-configuration"
CILOGON_SCOPE = "openid email profile org.cilogon.userinfo"
PROVIDER_NAME = "cilogon"
CALLBACK_ROUTE = "/auth/callback"
LOGIN_STATE_COOKIE = "tc_bench_login"
LOGIN_STATE_MAX_AGE_SECONDS = 600
DISPLAY_NAME_MIN_LENGTH = 2
DISPLAY_NAME_MAX_LENGTH = 60
FALLBACK_DISPLAY_NAME = "Submitter"

ClaimText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]


class LoginError(StrEnum):
    """Why a sign-in ended without a session (sent to the account page as a code)."""

    DENIED = "denied"  # the person cancelled, or the provider refused
    FAILED = "failed"  # the response did not verify (state, signature, nonce, expiry)
    UNAVAILABLE = "unavailable"  # the provider could not be reached
    NO_EMAIL = "no_email"  # the provider released no usable email address
    IDP_NOT_ALLOWED = "idp_not_allowed"
    EMAIL_NOT_ALLOWED = "email_not_allowed"
    EMAIL_IN_USE = "email_in_use"  # another sign-in already holds this address
    TOO_MANY_ACCOUNTS = "too_many_accounts"
    DISABLED = "disabled"


class LoginRefused(Exception):
    """A sign-in that verified but may not have a session."""

    def __init__(self, error: LoginError) -> None:
        """Refuse with ``error``."""
        super().__init__(error.value)
        self.error = error


class OidcConfig(BaseModel):
    """The OpenID Connect client registered with CILogon."""

    model_config = ConfigDict(frozen=True)

    client_id: str
    client_secret: SecretStr
    redirect_uri: str
    metadata_url: str = CILOGON_METADATA_URL
    scope: str = CILOGON_SCOPE

    @property
    def secure(self) -> bool:
        """True when the callback is served over HTTPS (the state cookie is then Secure)."""
        return urlsplit(self.redirect_uri).scheme == "https"

    @property
    def cookie_path(self) -> str:
        """The path the sign-in routes share, as the browser sees it."""
        return urlsplit(self.redirect_uri).path.rsplit("/", 1)[0]


def callback_url(public_url: str, api_prefix: str) -> str:
    """The callback URL to register with CILogon, for a service public at ``public_url``."""
    return f"{public_url.rstrip('/')}{api_prefix}{CALLBACK_ROUTE}"


class Identity(BaseModel):
    """Who signed in, read from a verified ID token."""

    model_config = ConfigDict(frozen=True)

    issuer: ClaimText
    subject: ClaimText
    email: EmailStr
    name: str | None = None
    idp: ClaimText | None = None
    idp_name: ClaimText | None = None

    @property
    def display_name(self) -> str:
        """A display name seeded from the claims: the name, else the email's local part."""
        for candidate in (self.name, self.email.partition("@")[0]):
            text = (candidate or "").strip()[:DISPLAY_NAME_MAX_LENGTH]
            if len(text) >= DISPLAY_NAME_MIN_LENGTH:
                return text
        return FALLBACK_DISPLAY_NAME


def identity_from_claims(claims: Mapping[str, Any]) -> Identity:
    """The :class:`Identity` in verified ID token ``claims``.

    Raises :class:`LoginRefused` with ``NO_EMAIL`` when the provider released no email
    address, and with ``FAILED`` when a claim has the wrong shape.
    """
    if not claims.get("email"):
        raise LoginRefused(LoginError.NO_EMAIL)
    given_and_family = " ".join(
        str(claims[key]) for key in ("given_name", "family_name") if claims.get(key)
    )
    name = claims.get("name") or given_and_family or None
    try:
        return Identity.model_validate(
            {
                "issuer": claims.get("iss"),
                "subject": claims.get("sub"),
                "email": claims["email"],
                "name": name,
                "idp": claims.get("idp") or None,
                "idp_name": claims.get("idp_name") or None,
            }
        )
    except ValidationError as error:
        # Claims come from outside; a malformed one is a failed sign-in, not a crash.
        raise LoginRefused(LoginError.FAILED) from error


def build_provider(config: OidcConfig, transport: Any = None) -> StarletteOAuth2App:
    """The Authlib client for ``config``.

    ``transport`` replaces the HTTP transport of every request the client makes
    (discovery, keys, token); tests pass one that answers as an identity provider.
    """
    client_kwargs: dict[str, Any] = {
        "scope": config.scope,
        "code_challenge_method": "S256",
        "timeout": 15.0,
    }
    if transport is not None:
        client_kwargs["transport"] = transport
    provider: StarletteOAuth2App = OAuth().register(
        name=PROVIDER_NAME,
        client_id=config.client_id,
        client_secret=config.client_secret.get_secret_value(),
        server_metadata_url=config.metadata_url,
        client_kwargs=client_kwargs,
    )
    return provider
