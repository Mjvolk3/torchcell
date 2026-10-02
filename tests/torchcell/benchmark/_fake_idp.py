# tests/torchcell/benchmark/_fake_idp.py
# [[tests.torchcell.benchmark._fake_idp]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/_fake_idp.py
"""An in-process OpenID Connect provider that answers the way CILogon does.

The benchmark service reaches its identity provider through one HTTP transport. In
the tests that transport is :meth:`FakeIdp.transport`, so discovery, the key set and
the token endpoint are all answered here, with no network. The provider is strict
where a real one is: the token endpoint checks the client's credentials, that the
code was issued and not yet used, that the redirect URI matches, and that the PKCE
verifier hashes to the challenge sent with the authorization request.

A test plays the browser. It reads the redirect ``GET /auth/login`` returns, hands it
to :meth:`FakeIdp.authorize` with the claims of the person signing in, and requests the
callback URL that comes back. Keyword overrides let a test issue a wrong ID token (a
stale nonce, another audience, another issuer, an expired token, a foreign signature).
"""

import base64
import hashlib
import json
import secrets
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx2
from joserfc import jwt
from joserfc.jwk import RSAKey

ISSUER = "https://idp.example"
CLIENT_ID = "cilogon:/client_id/test"
CLIENT_SECRET = "client-secret-for-tests"
UIUC = "urn:mace:incommon:uiuc.edu"
UIUC_NAME = "University of Illinois Urbana-Champaign"
GOOGLE = "http://google.com/accounts/o8/id"


def person(
    email: str | None = "alice@example.org",
    name: str | None = "Alice",
    sub: str | None = None,
    idp: str | None = UIUC,
    idp_name: str | None = UIUC_NAME,
) -> dict[str, Any]:
    """The identity claims of one person; ``sub`` defaults to one derived from the email."""
    claims = {
        "sub": sub or f"http://cilogon.org/serverA/users/{email}",
        "email": email,
        "name": name,
        "idp": idp,
        "idp_name": idp_name,
    }
    return {key: value for key, value in claims.items() if value is not None}


class FakeIdp:
    """One identity provider with one signing key and one registered client."""

    def __init__(self) -> None:
        """Generate the signing key; nothing has been issued yet."""
        self.key = RSAKey.generate_key(2048, parameters={"kid": "idp-key-1"})
        self.other_key = RSAKey.generate_key(2048, parameters={"kid": "idp-key-1"})
        self.grants: dict[str, dict[str, Any]] = {}
        self.down = False
        self.token_requests = 0

    @property
    def metadata_url(self) -> str:
        """The discovery document's URL."""
        return f"{ISSUER}/.well-known/openid-configuration"

    def transport(self) -> httpx2.MockTransport:
        """The HTTP transport the service uses to reach this provider."""
        return httpx2.MockTransport(self._handle)

    def authorize(self, location: str, claims: dict[str, Any], **overrides: Any) -> str:
        """Play the person's visit to the authorization endpoint.

        ``location`` is the redirect the service sent the browser to. Returns the
        callback URL (path and query) the provider sends the browser back to.
        """
        url = urlsplit(location)
        assert f"{url.scheme}://{url.netloc}{url.path}" == f"{ISSUER}/authorize"
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        assert query["response_type"] == "code"
        assert query["client_id"] == CLIENT_ID
        assert query["code_challenge_method"] == "S256"
        assert set(query["scope"].split()) >= {"openid", "email", "profile"}
        code = secrets.token_urlsafe(16)
        self.grants[code] = {
            "redirect_uri": query["redirect_uri"],
            "code_challenge": query["code_challenge"],
            "nonce": query["nonce"],
            "claims": claims,
            "overrides": overrides,
        }
        callback = urlsplit(query["redirect_uri"])
        return f"{callback.path}?{urlencode({'code': code, 'state': query['state']})}"

    def _id_token(self, grant: dict[str, Any]) -> str:
        now = int(time.time())
        overrides = grant["overrides"]
        claims = {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "iat": now,
            "exp": now + 300,
            "nonce": grant["nonce"],
            **grant["claims"],
            **{k: v for k, v in overrides.items() if k != "foreign_signature"},
        }
        key = self.other_key if overrides.get("foreign_signature") else self.key
        return jwt.encode({"alg": "RS256", "kid": "idp-key-1"}, claims, key)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        if self.down:
            raise httpx2.ConnectError("provider is down", request=request)
        path = request.url.path
        if path == "/.well-known/openid-configuration":
            return httpx2.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/authorize",
                    "token_endpoint": f"{ISSUER}/oauth2/token",
                    "jwks_uri": f"{ISSUER}/oauth2/certs",
                    "id_token_signing_alg_values_supported": ["RS256"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["client_secret_basic"],
                },
            )
        if path == "/oauth2/certs":
            return httpx2.Response(
                200, json={"keys": [self.key.as_dict(private=False)]}
            )
        if path == "/oauth2/token":
            return self._token(request)
        return httpx2.Response(404, json={"error": "not_found"})

    def _token(self, request: httpx2.Request) -> httpx2.Response:
        self.token_requests += 1
        basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        if request.headers.get("authorization") != f"Basic {basic}":
            return httpx2.Response(401, json={"error": "invalid_client"})
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        grant = self.grants.pop(form.get("code", ""), None)
        challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(form.get("code_verifier", "").encode()).digest()
            )
            .rstrip(b"=")
            .decode()
        )
        if (
            form.get("grant_type") != "authorization_code"
            or grant is None
            or form.get("redirect_uri") != grant["redirect_uri"]
            or challenge != grant["code_challenge"]
        ):
            return httpx2.Response(400, json={"error": "invalid_grant"})
        return httpx2.Response(
            200,
            content=json.dumps(
                {
                    "access_token": secrets.token_urlsafe(16),
                    "token_type": "Bearer",
                    "expires_in": 300,
                    "id_token": self._id_token(grant),
                }
            ),
            headers={"content-type": "application/json"},
        )
