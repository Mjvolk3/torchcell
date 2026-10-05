# tests/torchcell/benchmark/test_app.py
# [[tests.torchcell.benchmark.test_app]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_app.py
"""``torchcell.benchmark.app`` (tc-bench) end to end, in process.

The app runs on a SQLite file, the ``toy-fitness`` bundle of ``conftest.py``, the
in-process identity provider of ``_fake_idp.py`` in place of CILogon, and a settable
clock (the quota and the sign-in codes read it; session tokens use the wall clock).
Scores are asserted against the hand-worked values: the labels themselves score Pearson
1 on both splits, and the validation predictions 2, 1, 3, 4 against the labels
1, 2, 3, 4 score Pearson 0.8.

Covered: the sign-in sequence (redirect with state, nonce and PKCE; callback; one-time
code; bearer token); personal API tokens (shown once, stored hashed, able to read and
submit but not to manage the account, revocable, capped per account); each way a
sign-in is refused (a forged or replayed state, an ID token
with the wrong nonce, audience, issuer, expiry or signature, a provider that is down, a
cancelled sign-in, a missing email, a provider or domain the policy excludes, a second
identity on one address, too many new accounts from one address, a disabled account);
the profile route; the public dataset routes and their sha256 header; a scored
submission, its archive and its board row; a rejected submission and its reasons; the
quota (one hour apart, three per 24 hours, rejected attempts counted); the upload size
guard; the integrity flag; admin verify, withdraw, approve, disable and baselines; and
``BenchServerConfig.from_env``.
"""

import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from sqlalchemy import create_engine, select

from tests.torchcell.benchmark._fake_idp import (
    CLIENT_ID,
    CLIENT_SECRET,
    GOOGLE,
    UIUC,
    UIUC_NAME,
    FakeIdp,
    person,
)
from torchcell.api_keys import ApiKeys
from torchcell.benchmark import app as app_module
from torchcell.benchmark.app import API_PREFIX, BenchServerConfig, create_app
from torchcell.benchmark.db import (
    ApiToken,
    LoginCode,
    User,
    init_schema,
    make_session_factory,
)
from torchcell.benchmark.oidc import CILOGON_METADATA_URL, OidcConfig
from torchcell.benchmark.security import AccountPolicy

SLUG = "toy-fitness"
ADMIN = {"X-API-Key": "admin-key-123"}
SERVICE = "https://bench.example"
ACCOUNT_URL = "https://site.example/benchmark/account/"
T0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
METADATA: dict[str, Any] = {
    "method_name": "ridge on one-hot",
    "description": "Ridge regression on a one-hot gene encoding.",
    "model_family": "ridge",
    "encoding": "one-hot gene",
    "uses_external_data": False,
}
Labels = dict[tuple[str, str], float]
Csv = Callable[[Mapping[tuple[str, str], float]], bytes]


class Clock:
    """A settable clock."""

    def __init__(self) -> None:
        """Start at ``T0``."""
        self.now = T0

    def __call__(self) -> datetime:
        """The time the clock is set to."""
        return self.now


class Bench:
    """The app under test with its collaborators."""

    def __init__(self, tmp_path: Path, datasets_root: Path, **overrides: Any) -> None:
        """Build the app on a SQLite file under ``tmp_path``."""
        self.submissions_root = tmp_path / "submissions"
        self.submissions_root.mkdir()
        self.idp = FakeIdp()
        self.config = BenchServerConfig(
            database_url=SecretStr("postgresql+psycopg://unused"),
            datasets_root=datasets_root,
            submissions_root=self.submissions_root,
            jwt_secret=SecretStr("j" * 40),
            admin_keys=ApiKeys.from_pairs("ops:admin-key-123"),
            account_url=ACCOUNT_URL,
            cors_origins=("https://site.example",),
            oidc=OidcConfig(
                client_id=CLIENT_ID,
                client_secret=SecretStr(CLIENT_SECRET),
                redirect_uri=f"{SERVICE}{API_PREFIX}/auth/callback",
                metadata_url=self.idp.metadata_url,
            ),
            **overrides,
        )
        self.engine = create_engine(
            f"sqlite:///{tmp_path / 'bench.db'}",
            connect_args={"check_same_thread": False},
        )
        init_schema(self.engine)
        self.sessions = make_session_factory(self.engine)
        self.clock = Clock()
        self.app = create_app(
            self.config,
            engine=self.engine,
            oidc_transport=self.idp.transport(),
            clock=self.clock,
        )
        self.client = TestClient(self.app, base_url=SERVICE)

    def url(self, path: str) -> str:
        """The full URL path of an API route."""
        return f"{API_PREFIX}{path}"

    def start(self, client: TestClient | None = None) -> str:
        """Begin a sign-in; return the provider URL the browser is sent to."""
        response = (client or self.client).get(
            self.url("/auth/login"), follow_redirects=False
        )
        assert response.status_code == 302
        return str(response.headers["location"])

    def sign_in(
        self,
        claims: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **overrides: Any,
    ) -> dict[str, str]:
        """Run a whole sign-in as ``claims``; return the account page's URL fragment."""
        callback = self.idp.authorize(self.start(), claims or person(), **overrides)
        return self.finish(callback, headers)

    def finish(
        self, callback: str, headers: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Request ``callback``; return the fragment of the account page redirect."""
        response = self.client.get(callback, headers=headers, follow_redirects=False)
        assert response.status_code == 303
        target = urlsplit(response.headers["location"])
        assert f"{target.scheme}://{target.netloc}{target.path}" == ACCOUNT_URL
        assert target.query == ""
        return {key: values[0] for key, values in parse_qs(target.fragment).items()}

    def exchange(self, code: str) -> Any:
        """Trade a sign-in code for a token (the response, whatever its status)."""
        return self.client.post(self.url("/auth/exchange"), json={"code": code})

    def register(
        self, email: str = "alice@example.org", name: str = "Alice"
    ) -> dict[str, str]:
        """Sign in as a new person; return the bearer header."""
        fragment = self.sign_in(person(email, name))
        assert set(fragment) == {"login_code"}
        token = self.exchange(fragment["login_code"])
        assert token.status_code == 200
        return {"Authorization": f"Bearer {token.json()['access_token']}"}

    def submit(
        self,
        headers: dict[str, str],
        raw: bytes,
        metadata: dict[str, Any] | None = None,
        path: str = "/submissions",
    ) -> Any:
        """Upload ``raw`` as a submission (or a baseline, by ``path``)."""
        return self.client.post(
            self.url(path),
            headers=headers,
            data={"dataset": SLUG, "metadata": json.dumps(metadata or METADATA)},
            files={"predictions": ("predictions.csv", raw, "text/csv")},
        )


@pytest.fixture
def bench(tmp_path: Path, datasets_root: Path) -> Bench:
    return Bench(tmp_path, datasets_root)


# ------------------------------------------------------------------------ service


def test_health_and_schema(bench: Bench) -> None:
    assert bench.client.get(bench.url("/health")).json() == {
        "status": "ok",
        "n_datasets": 1,
        "tier": "production",
        "build": None,
    }
    schema = bench.client.get(bench.url("/submission-schema")).json()
    assert schema["columns"] == ["record_id", "split", "target", "prediction"]
    assert set(schema) == {"columns", "prediction_row", "metadata"}


def test_security_headers_and_cors(bench: Bench) -> None:
    response = bench.client.get(
        bench.url("/health"), headers={"Origin": "https://site.example"}
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["access-control-allow-origin"] == "https://site.example"
    other = bench.client.get(
        bench.url("/health"), headers={"Origin": "https://evil.example"}
    )
    assert "access-control-allow-origin" not in other.headers


# --------------------------------------------------------------------------- auth


def test_sign_in_sequence(bench: Bench) -> None:
    location = bench.start()
    provider = urlsplit(location)
    query = {key: values[0] for key, values in parse_qs(provider.query).items()}
    assert f"{provider.scheme}://{provider.netloc}{provider.path}" == (
        "https://idp.example/authorize"
    )
    assert query["redirect_uri"] == "https://bench.example/api/v1/auth/callback"
    assert query["scope"] == "openid email profile org.cilogon.userinfo"
    assert query["code_challenge_method"] == "S256"
    assert len(query["state"]) >= 20 and len(query["nonce"]) >= 20
    assert len(query["code_challenge"]) == 43  # base64url of a sha256

    fragment = bench.finish(bench.idp.authorize(location, person("Alice@Example.org")))
    assert set(fragment) == {"login_code"}
    token = bench.exchange(fragment["login_code"])
    assert token.status_code == 200
    assert token.json()["token_type"] == "bearer"
    me = bench.client.get(
        bench.url("/auth/me"),
        headers={"Authorization": f"Bearer {token.json()['access_token']}"},
    ).json()
    assert me["email"] == "alice@example.org"
    assert me["display_name"] == "Alice"
    assert me["identity_provider"] == UIUC_NAME
    assert me["approved"] is True
    assert me["created_at"] == "2026-10-01T12:00:00Z"
    assert "email_verified" not in me
    with bench.sessions() as session:
        user = session.scalars(select(User)).one()
        assert (user.oidc_issuer, user.idp) == ("https://idp.example", UIUC)
        assert user.oidc_subject == "http://cilogon.org/serverA/users/Alice@Example.org"
        assert user.last_login_at == T0


def test_login_state_cookie_is_scoped_and_signed(bench: Bench) -> None:
    response = bench.client.get(bench.url("/auth/login"), follow_redirects=False)
    cookie = response.headers["set-cookie"]
    name, _, value = cookie.partition(";")[0].partition("=")
    assert name == "tc_bench_login"
    attributes = {part.strip().lower() for part in cookie.split(";")[1:]}
    assert {"httponly", "secure", "samesite=lax", "max-age=600"} <= attributes
    assert "path=/api/v1/auth" in attributes
    # The cookie is signed with a key derived from the session secret, not with it.
    assert bench.config.jwt_secret.get_secret_value() not in value
    # No other route sets a cookie: the session itself is a bearer token.
    assert "set-cookie" not in bench.client.get(bench.url("/health")).headers


def test_sign_in_code_works_once_and_expires(bench: Bench) -> None:
    code = bench.sign_in()["login_code"]
    assert bench.exchange(code).status_code == 200
    reused = bench.exchange(code)
    assert reused.status_code == 400
    assert reused.json() == {"detail": "invalid or expired code"}
    assert bench.exchange("never-issued").status_code == 400

    late = bench.sign_in()["login_code"]
    bench.clock.now = T0 + timedelta(minutes=2)
    assert bench.exchange(late).status_code == 400
    with bench.sessions() as session:
        stored = session.scalars(select(LoginCode.code_sha256)).all()
        assert len(stored) == 2 and code not in stored and late not in stored
    # The next sign-in clears this account's expired codes, spent or not.
    bench.sign_in()
    with bench.sessions() as session:
        codes = session.scalars(select(LoginCode)).all()
        assert [(c.used_at, c.expires_at) for c in codes] == [
            (None, T0 + timedelta(minutes=4))
        ]


def test_second_sign_in_reuses_the_account(bench: Bench) -> None:
    first = bench.register()
    bench.clock.now = T0 + timedelta(days=1)
    second = bench.register()
    ids = {
        bench.client.get(bench.url("/auth/me"), headers=h).json()["user_id"]
        for h in (first, second)
    }
    assert len(ids) == 1
    with bench.sessions() as session:
        user = session.scalars(select(User)).one()
        assert (user.created_at, user.last_login_at) == (T0, T0 + timedelta(days=1))


def test_callback_refuses_a_forged_or_replayed_state(bench: Bench) -> None:
    location = bench.start()
    callback = bench.idp.authorize(location, person())
    forged = callback.replace("state=", "state=x")
    assert bench.finish(forged) == {"login_error": "failed"}
    assert bench.idp.token_requests == 0  # refused before the code was spent

    # A callback that arrives in a browser that never started the sign-in is refused:
    # the state is only valid together with that browser's cookie.
    other_browser = TestClient(bench.app, base_url=SERVICE)
    stolen = other_browser.get(callback, follow_redirects=False)
    assert stolen.headers["location"] == f"{ACCOUNT_URL}#login_error=failed"

    callback = bench.idp.authorize(bench.start(), person())
    assert set(bench.finish(callback)) == {"login_code"}
    assert bench.finish(callback) == {"login_error": "failed"}  # replay
    with bench.sessions() as session:
        assert len(session.scalars(select(User)).all()) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"nonce": "a-nonce-from-another-sign-in"},
        {"aud": "cilogon:/client_id/someone-else"},
        {"iss": "https://evil.example"},
        {"exp": 1_000_000_000},
        {"foreign_signature": True},
    ],
    ids=["nonce", "audience", "issuer", "expired", "signature"],
)
def test_callback_refuses_an_id_token_that_does_not_verify(
    bench: Bench, override: dict[str, Any]
) -> None:
    assert bench.sign_in(person(), **override) == {"login_error": "failed"}
    assert bench.idp.token_requests == 1  # the token was fetched, then refused
    with bench.sessions() as session:
        assert session.scalars(select(User)).all() == []


def test_callback_reports_a_cancelled_sign_in(bench: Bench) -> None:
    state = parse_qs(urlsplit(bench.start()).query)["state"][0]
    cancelled = bench.finish(
        bench.url(f"/auth/callback?error=access_denied&state={state}")
    )
    assert cancelled == {"login_error": "denied"}
    other = bench.finish(bench.url(f"/auth/callback?error=server_error&state={state}"))
    assert other == {"login_error": "failed"}


def test_provider_outage_is_reported_not_raised(
    bench: Bench, tmp_path: Path, datasets_root: Path
) -> None:
    callback = bench.idp.authorize(bench.start(), person())
    bench.idp.down = True
    assert bench.finish(callback) == {"login_error": "unavailable"}

    (tmp_path / "fresh").mkdir()
    fresh = Bench(tmp_path / "fresh", datasets_root)
    fresh.idp.down = True  # discovery itself fails on the first sign-in
    response = fresh.client.get(fresh.url("/auth/login"), follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"{ACCOUNT_URL}#login_error=unavailable"


def test_sign_in_without_an_email_is_refused(bench: Bench) -> None:
    assert bench.sign_in(person(email=None, sub="orcid-1")) == {
        "login_error": "no_email"
    }
    assert bench.sign_in(person(email="not-an-address", sub="orcid-2")) == {
        "login_error": "failed"
    }
    with bench.sessions() as session:
        assert session.scalars(select(User)).all() == []


def test_one_account_per_canonical_address(bench: Bench) -> None:
    bench.register("alice@gmail.com")
    # The same mailbox through another identity is refused, not merged.
    second = bench.sign_in(
        person("a.lice+two@googlemail.com", sub="google-oauth2|123", idp=GOOGLE)
    )
    assert second == {"login_error": "email_in_use"}
    with bench.sessions() as session:
        assert session.scalars(select(User.email_canonical)).all() == [
            "alice@gmail.com"
        ]


def test_new_accounts_per_address_are_limited(bench: Bench) -> None:
    for i in range(5):
        bench.register(f"user{i}@example.org")
    assert bench.sign_in(person("user5@example.org")) == {
        "login_error": "too_many_accounts"
    }
    # an existing account still signs in from that address
    assert set(bench.sign_in(person("user0@example.org"))) == {"login_code"}
    bench.clock.now = T0 + timedelta(hours=24, seconds=1)
    assert set(bench.sign_in(person("user5@example.org"))) == {"login_code"}


def test_account_policy_limits_who_can_register(
    tmp_path: Path, datasets_root: Path
) -> None:
    bench = Bench(
        tmp_path,
        datasets_root,
        account_policy=AccountPolicy(
            blocked_domains=frozenset({"mailinator.com"}),
            allowed_idps=frozenset({UIUC}),
        ),
    )
    assert bench.sign_in(person("x@mailinator.com")) == {
        "login_error": "email_not_allowed"
    }
    assert bench.sign_in(person("bob@example.org", idp=GOOGLE)) == {
        "login_error": "idp_not_allowed"
    }
    assert bench.sign_in(person("carol@example.org", idp=None)) == {
        "login_error": "idp_not_allowed"
    }
    assert set(bench.sign_in(person("dana@example.org"))) == {"login_code"}
    with bench.sessions() as session:
        assert session.scalars(select(User.email)).all() == ["dana@example.org"]


def test_display_name_comes_from_the_claims_and_can_be_edited(bench: Bench) -> None:
    fragment = bench.sign_in(person("grace.hopper@example.org", name=None))
    token = bench.exchange(fragment["login_code"]).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    me = bench.client.get(bench.url("/auth/me"), headers=headers).json()
    assert (me["display_name"], me["affiliation"]) == ("grace.hopper", None)

    updated = bench.client.post(
        bench.url("/auth/profile"),
        headers=headers,
        json={"display_name": "  Grace Hopper ", "affiliation": "Navy"},
    )
    assert updated.status_code == 200
    assert (updated.json()["display_name"], updated.json()["affiliation"]) == (
        "Grace Hopper",
        "Navy",
    )
    history = bench.client.get(bench.url(f"/users/{me['user_id']}/submissions")).json()
    assert history["user"] == {
        "user_id": me["user_id"],
        "display_name": "Grace Hopper",
        "affiliation": "Navy",
        "identity_provider": UIUC_NAME,
        "created_at": "2026-10-01T12:00:00Z",
    }
    for body, field in [
        ({"display_name": "G"}, "display_name"),
        ({"display_name": "Grace", "email": "x@example.org"}, "email"),
        ({"display_name": "Grace", "approved": True}, "approved"),
    ]:
        refused = bench.client.post(
            bench.url("/auth/profile"), headers=headers, json=body
        )
        assert refused.status_code == 422
        assert refused.json()["detail"][0]["loc"] == ["body", field]
    assert (
        bench.client.post(
            bench.url("/auth/profile"), json={"display_name": "Grace"}
        ).status_code
        == 401
    )


def test_routes_that_need_a_session(bench: Bench) -> None:
    for method, path in [
        ("GET", "/auth/me"),
        ("GET", "/quota"),
        ("GET", "/submissions/mine"),
    ]:
        response = bench.client.request(method, bench.url(path))
        assert response.status_code == 401
        assert response.json() == {
            "detail": "sign in on the account page to use this route"
        }
        assert response.headers["www-authenticate"] == "Bearer"
    forged = bench.client.get(
        bench.url("/auth/me"), headers={"Authorization": "Bearer x.y.z"}
    )
    assert forged.status_code == 401


# --------------------------------------------------------------------- API tokens


def _new_token(bench: Bench, headers: dict[str, str], name: str = "laptop") -> Any:
    return bench.client.post(
        bench.url("/auth/tokens"), headers=headers, json={"name": name}
    )


def test_api_token_is_shown_once_and_stored_as_a_hash(bench: Bench) -> None:
    headers = bench.register()
    response = _new_token(bench, headers, "  laptop  ")
    assert response.status_code == 201
    created = response.json()
    assert set(created) == {
        "token_id",
        "name",
        "hint",
        "created_at",
        "expires_at",
        "last_used_at",
        "token",
    }
    token = created["token"]
    assert token.startswith("tcb_")
    assert (created["name"], created["hint"]) == ("laptop", token[:12])
    assert created["created_at"] == "2026-10-01T12:00:00Z"
    assert created["expires_at"] == "2026-12-30T12:00:00Z"  # the 90-day default
    assert created["last_used_at"] is None

    with bench.sessions() as session:
        row = session.scalars(select(ApiToken)).one()
        assert row.token_sha256 == hashlib.sha256(token.encode()).hexdigest()
        assert token not in (row.name, row.token_hint, row.token_sha256)

    listed = bench.client.get(bench.url("/auth/tokens"), headers=headers).json()
    assert listed == [{k: v for k, v in created.items() if k != "token"}]


def test_api_token_submits_and_reads_but_does_not_manage_the_account(
    bench: Bench, labels: Labels, to_csv: Csv
) -> None:
    session_headers = bench.register()
    token = _new_token(bench, session_headers).json()["token"]
    api = {"Authorization": f"Bearer {token}"}

    me = bench.client.get(bench.url("/auth/me"), headers=api)
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.org"
    assert bench.client.get(bench.url("/quota"), headers=api).json()["remaining"] == 3

    bench.clock.now = T0 + timedelta(minutes=5)
    scored = bench.submit(api, to_csv(labels))
    assert scored.status_code == 201
    assert scored.json()["status"] == "provisional"
    mine = bench.client.get(bench.url("/submissions/mine"), headers=api).json()
    assert [m["submission_id"] for m in mine] == [scored.json()["submission_id"]]
    # The quota is the account's: the attempt made with the token is counted for the
    # session too.
    quota = bench.client.get(bench.url("/quota"), headers=session_headers).json()
    assert quota["used_in_window"] == 1

    listed = bench.client.get(bench.url("/auth/tokens"), headers=session_headers).json()
    assert listed[0]["last_used_at"] == "2026-10-01T12:05:00Z"

    for method, path, body in [
        ("GET", "/auth/tokens", None),
        ("POST", "/auth/tokens", {"name": "second"}),
        ("POST", f"/auth/tokens/{listed[0]['token_id']}/revoke", None),
        ("POST", "/auth/profile", {"display_name": "Mallory"}),
    ]:
        refused = bench.client.request(method, bench.url(path), headers=api, json=body)
        assert refused.status_code == 401
        assert refused.json() == {
            "detail": "sign in on the account page to use this route"
        }


def test_revoked_or_unknown_api_token_is_refused(bench: Bench) -> None:
    headers = bench.register()
    created = _new_token(bench, headers).json()
    api = {"Authorization": f"Bearer {created['token']}"}
    revoke = bench.url(f"/auth/tokens/{created['token_id']}/revoke")

    other = bench.register("bob@example.org", "Bob")
    assert bench.client.post(revoke, headers=other).status_code == 404
    assert bench.client.get(bench.url("/quota"), headers=api).status_code == 200

    revoked = bench.client.post(revoke, headers=headers)
    assert revoked.json() == {"message": "API token laptop revoked"}
    assert bench.client.post(revoke, headers=headers).status_code == 404
    assert bench.client.get(bench.url("/auth/tokens"), headers=headers).json() == []
    for credential in (created["token"], "tcb_" + "x" * 43):
        response = bench.client.get(
            bench.url("/quota"), headers={"Authorization": f"Bearer {credential}"}
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid, revoked or expired API token"}
    with bench.sessions() as session:  # a revoked token leaves no row behind
        assert session.scalars(select(ApiToken)).all() == []


def test_api_token_expires_and_its_row_is_removed(bench: Bench) -> None:
    headers = bench.register()
    short = bench.client.post(
        bench.url("/auth/tokens"),
        headers=headers,
        json={"name": "short", "expires_in_days": 1},
    ).json()
    assert short["expires_at"] == "2026-10-02T12:00:00Z"
    long = bench.client.post(
        bench.url("/auth/tokens"),
        headers=headers,
        json={"name": "long", "expires_in_days": 365},
    ).json()
    assert long["expires_at"] == "2027-10-01T12:00:00Z"
    too_long = bench.client.post(
        bench.url("/auth/tokens"),
        headers=headers,
        json={"name": "forever", "expires_in_days": 366},
    )
    assert too_long.status_code == 422
    assert too_long.json() == {"detail": "a token lasts at most 365 days"}
    assert (
        bench.client.post(
            bench.url("/auth/tokens"),
            headers=headers,
            json={"name": "zero", "expires_in_days": 0},
        ).status_code
        == 422
    )

    api = {"Authorization": f"Bearer {short['token']}"}
    bench.clock.now = T0 + timedelta(hours=23, minutes=59)
    assert bench.client.get(bench.url("/quota"), headers=api).status_code == 200
    bench.clock.now = T0 + timedelta(days=1)
    expired = bench.client.get(bench.url("/quota"), headers=api)
    assert expired.status_code == 401
    assert expired.json() == {"detail": "invalid, revoked or expired API token"}

    # Listing drops the expired row and keeps the live one.
    listed = bench.client.get(bench.url("/auth/tokens"), headers=headers).json()
    assert [t["name"] for t in listed] == ["long"]
    with bench.sessions() as session:
        assert [row.name for row in session.scalars(select(ApiToken))] == ["long"]


def test_expired_tokens_do_not_count_toward_the_cap(
    tmp_path: Path, datasets_root: Path
) -> None:
    bench = Bench(tmp_path, datasets_root, max_api_tokens=1)
    headers = bench.register()
    body = {"name": "one", "expires_in_days": 1}
    created = bench.client.post(bench.url("/auth/tokens"), headers=headers, json=body)
    assert created.status_code == 201
    again = bench.client.post(bench.url("/auth/tokens"), headers=headers, json=body)
    assert again.status_code == 409
    bench.clock.now = T0 + timedelta(days=2)
    assert (
        bench.client.post(
            bench.url("/auth/tokens"), headers=headers, json=body
        ).status_code
        == 201
    )


def test_api_tokens_per_account_are_capped_and_named(
    tmp_path: Path, datasets_root: Path
) -> None:
    bench = Bench(tmp_path, datasets_root, max_api_tokens=2)
    headers = bench.register()
    first = _new_token(bench, headers, "one").json()
    assert _new_token(bench, headers, "two").status_code == 201
    full = _new_token(bench, headers, "three")
    assert full.status_code == 409
    assert full.json() == {
        "detail": "an account holds at most 2 API tokens; revoke one first"
    }
    bench.client.post(
        bench.url(f"/auth/tokens/{first['token_id']}/revoke"), headers=headers
    )
    assert _new_token(bench, headers, "three").status_code == 201
    for bad in ({"name": ""}, {"name": "n" * 61}, {}, {"name": "ok", "extra": 1}):
        response = bench.client.post(
            bench.url("/auth/tokens"), headers=headers, json=bad
        )
        assert response.status_code == 422


def test_disabled_account_loses_its_api_tokens(bench: Bench) -> None:
    headers = bench.register()
    token = _new_token(bench, headers).json()["token"]
    user_id = bench.client.get(bench.url("/auth/me"), headers=headers).json()["user_id"]
    bench.client.post(bench.url(f"/admin/users/{user_id}/disable"), headers=ADMIN)
    response = bench.client.get(
        bench.url("/quota"), headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


# ----------------------------------------------------------------------- datasets


def test_dataset_routes(bench: Bench, datasets_root: Path) -> None:
    listing = bench.client.get(bench.url("/datasets")).json()
    assert [d["slug"] for d in listing] == [SLUG]
    one = bench.client.get(bench.url(f"/datasets/{SLUG}")).json()
    assert one == listing[0]
    assert (one["n_train"], one["n_val"], one["n_test"]) == (2, 4, 4)
    assert one["targets"] == ["fitness"]
    assert one["primary_metric"] == "pearson"
    assert "labels_sha256" not in one

    template = bench.client.get(bench.url(f"/datasets/{SLUG}/template.csv"))
    assert template.content == (datasets_root / SLUG / "template.csv").read_bytes()
    assert template.headers["x-artifact-sha256"] == one["template_sha256"]
    assert template.headers["content-type"].startswith("text/csv")
    splits = bench.client.get(bench.url(f"/datasets/{SLUG}/splits.csv"))
    assert splits.headers["x-artifact-sha256"] == one["splits_sha256"]
    assert splits.text.splitlines()[0] == "record_id,split"

    for path in ["/datasets/nope", "/datasets/nope/template.csv", "/leaderboard/nope"]:
        missing = bench.client.get(bench.url(path))
        assert missing.status_code == 404
        assert missing.json() == {"detail": "unknown benchmark dataset"}
    # the labels are not reachable under any dataset route
    assert (
        bench.client.get(bench.url(f"/datasets/{SLUG}/labels.csv")).status_code == 404
    )


# -------------------------------------------------------------------- submissions


def test_scored_submission(bench: Bench, labels: Labels, to_csv: Csv) -> None:
    headers = bench.register()
    raw = to_csv(labels)
    response = bench.submit(headers, raw)
    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "provisional"
    assert result["dataset_slug"] == SLUG
    assert result["method_name"] == "ridge on one-hot"
    assert result["rejection_reasons"] == []
    assert result["flags"] == []
    assert result["submitted_at"] == "2026-10-01T12:00:00Z"
    for split in ("val", "test"):
        assert result[split]["n_records"] == 4
        assert result[split]["macro"] == {
            "pearson": pytest.approx(1.0),
            "spearman": pytest.approx(1.0),
            "mse": 0.0,
            "mae": 0.0,
            "r2": 1.0,
        }
        assert list(result[split]["per_target"]) == ["fitness"]

    archive = (
        bench.submissions_root / SLUG / "2026" / "10" / f"{result['submission_id']}.zip"
    )
    assert archive.is_file()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["archive_sha256"]
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as zipped:
        assert zipped.namelist() == ["metadata.json", "predictions.csv", "result.json"]
        assert zipped.read("predictions.csv") == raw
        assert json.loads(zipped.read("result.json"))["val"]["macro"]["mse"] == 0.0

    board = bench.client.get(bench.url(f"/leaderboard/{SLUG}")).json()
    assert len(board) == 1
    row = board[0]
    assert row["display_name"] == "Alice"
    assert row["status"] == "provisional"
    assert (row["model_family"], row["encoding"]) == ("ridge", "one-hot gene")
    assert row["is_baseline"] is False
    assert "dataset_slug" not in row
    assert (
        bench.client.get(bench.url(f"/leaderboard/{SLUG}?verified_only=true")).json()
        == []
    )

    mine = bench.client.get(bench.url("/submissions/mine"), headers=headers).json()
    assert [m["submission_id"] for m in mine] == [result["submission_id"]]


def test_rejected_submission_lists_reasons_and_is_not_archived(
    bench: Bench, labels: Labels, to_csv: Csv
) -> None:
    headers = bench.register()
    del labels[("s4", "fitness")]
    response = bench.submit(headers, to_csv(labels))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["status"] == "rejected"
    assert detail["rejection_reasons"] == [
        "1 template pairs have no valid row, for example (s4, fitness)"
    ]
    assert detail["val"] is None and detail["test"] is None
    assert detail["archive_sha256"] is None
    assert list(bench.submissions_root.iterdir()) == []
    assert bench.client.get(bench.url(f"/leaderboard/{SLUG}")).json() == []
    mine = bench.client.get(bench.url("/submissions/mine"), headers=headers).json()
    assert [m["status"] for m in mine] == ["rejected"]


def test_bad_metadata_is_a_rejection(bench: Bench, labels: Labels, to_csv: Csv) -> None:
    headers = bench.register()
    response = bench.submit(headers, to_csv(labels), {**METADATA, "test_pearson": 0.99})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["rejection_reasons"] == [
        "metadata.test_pearson: Extra inputs are not permitted"
    ]
    assert detail["method_name"] == "(metadata not parsed)"


def test_quota_gap_and_window(bench: Bench, labels: Labels, to_csv: Csv) -> None:
    headers = bench.register()
    raw = to_csv(labels)
    fresh = bench.client.get(bench.url("/quota"), headers=headers).json()
    assert fresh == {
        "max_per_window": 3,
        "window_hours": 24.0,
        "min_gap_minutes": 60.0,
        "used_in_window": 0,
        "remaining": 3,
        "next_allowed_at": None,
    }

    # attempt 1 at T0 is rejected, and still counts
    assert (
        bench.submit(headers, b"record_id,split,target,prediction\n").status_code == 422
    )
    too_soon = bench.submit(headers, raw)
    assert too_soon.status_code == 429
    assert too_soon.json() == {
        "detail": {
            "message": "submissions must be 60 minutes apart",
            "next_allowed_at": "2026-10-01T13:00:00+00:00",
        }
    }

    bench.clock.now = T0 + timedelta(hours=1)
    assert bench.submit(headers, raw).status_code == 201
    bench.clock.now = T0 + timedelta(hours=2)
    assert bench.submit(headers, raw).status_code == 201

    bench.clock.now = T0 + timedelta(hours=5)
    spent = bench.submit(headers, raw)
    assert spent.status_code == 429
    assert spent.json() == {
        "detail": {
            "message": "3 submissions are allowed per 24 hours",
            "next_allowed_at": "2026-10-02T12:00:00+00:00",
        }
    }
    quota = bench.client.get(bench.url("/quota"), headers=headers).json()
    assert (quota["used_in_window"], quota["remaining"]) == (3, 0)
    assert quota["next_allowed_at"] == "2026-10-02T12:00:00Z"
    # a refused request is not an attempt: still three rows
    mine = bench.client.get(bench.url("/submissions/mine"), headers=headers).json()
    assert [m["status"] for m in mine] == ["provisional", "provisional", "rejected"]

    bench.clock.now = T0 + timedelta(hours=24)
    assert bench.submit(headers, raw).status_code == 201


def test_quota_is_per_account(bench: Bench, labels: Labels, to_csv: Csv) -> None:
    alice = bench.register("alice@example.org", "Alice")
    bob = bench.register("bob@example.org", "Bob")
    assert bench.submit(alice, to_csv(labels)).status_code == 201
    assert bench.submit(bob, to_csv(labels)).status_code == 201
    assert bench.submit(alice, to_csv(labels)).status_code == 429


def test_submission_needs_a_session_and_a_known_dataset(
    bench: Bench, labels: Labels, to_csv: Csv
) -> None:
    assert bench.submit({}, to_csv(labels)).status_code == 401
    headers = bench.register()
    unknown = bench.client.post(
        bench.url("/submissions"),
        headers=headers,
        data={"dataset": "nope", "metadata": json.dumps(METADATA)},
        files={"predictions": ("p.csv", to_csv(labels), "text/csv")},
    )
    assert unknown.status_code == 404
    # neither request reached the grader, so no attempt was recorded
    assert (
        bench.client.get(bench.url("/submissions/mine"), headers=headers).json() == []
    )


def test_upload_size_guard(tmp_path: Path, datasets_root: Path) -> None:
    bench = Bench(tmp_path, datasets_root, max_upload_bytes=1024)
    headers = bench.register()
    huge = b"record_id,split,target,prediction\n" + b"x" * (
        1024 + 16 * 1024 + 64 * 1024
    )
    response = bench.submit(headers, huge)
    assert response.status_code == 413
    assert response.json() == {
        "detail": f"upload exceeds {1024 + 16 * 1024 + 64 * 1024} bytes"
    }
    # within the request limit but over the file limit: a recorded rejection
    over_file_limit = bench.submit(
        headers, b"record_id,split,target,prediction\n" + b"x" * 2000
    )
    assert over_file_limit.status_code == 422
    assert over_file_limit.json()["detail"]["rejection_reasons"] == [
        "predictions file exceeds 1024 bytes"
    ]


def test_test_exceeding_validation_is_flagged(
    bench: Bench, labels: Labels, to_csv: Csv
) -> None:
    headers = bench.register()
    predictions = {**labels, ("v1", "fitness"): 2.0, ("v2", "fitness"): 1.0}
    result = bench.submit(headers, to_csv(predictions)).json()
    assert result["val"]["macro"]["pearson"] == pytest.approx(0.8)
    assert result["test"]["macro"]["pearson"] == pytest.approx(1.0)
    assert result["flags"] == ["test_exceeds_val"]
    board = bench.client.get(bench.url(f"/leaderboard/{SLUG}")).json()
    assert board[0]["flags"] == ["test_exceeds_val"]


def test_board_is_sorted_by_test_score_and_history_is_public(
    bench: Bench, labels: Labels, to_csv: Csv
) -> None:
    alice = bench.register("alice@example.org", "Alice")
    bob = bench.register("bob@example.org", "Bob")
    worse_test = {**labels, ("s1", "fitness"): 1.0, ("s2", "fitness"): 0.5}
    bench.submit(alice, to_csv(worse_test))
    bench.submit(bob, to_csv(labels))
    board = bench.client.get(bench.url(f"/leaderboard/{SLUG}")).json()
    assert [row["display_name"] for row in board] == ["Bob", "Alice"]
    assert board[0]["test"]["macro"]["pearson"] > board[1]["test"]["macro"]["pearson"]

    history = bench.client.get(
        bench.url(f"/users/{board[1]['user_id']}/submissions")
    ).json()
    assert history["user"]["display_name"] == "Alice"
    assert set(history["user"]) == {
        "user_id",
        "display_name",
        "affiliation",
        "identity_provider",
        "created_at",
    }
    assert [s["dataset_slug"] for s in history["submissions"]] == [SLUG]
    assert bench.client.get(bench.url("/users/nobody/submissions")).status_code == 404


# -------------------------------------------------------------------------- admin


def test_admin_routes_need_a_key(bench: Bench) -> None:
    for path in [
        "/admin/submissions/x/verify",
        "/admin/submissions/x/withdraw",
        "/admin/users/x/approve",
        "/admin/users/x/disable",
    ]:
        for headers in ({}, {"X-API-Key": "wrong"}):
            response = bench.client.post(bench.url(path), headers=headers, json={})
            assert response.status_code == 401
            assert response.json() == {"detail": "invalid or missing admin key"}


def test_verify_then_withdraw(bench: Bench, labels: Labels, to_csv: Csv) -> None:
    headers = bench.register()
    submission_id = bench.submit(headers, to_csv(labels)).json()["submission_id"]
    verify_url = bench.url(f"/admin/submissions/{submission_id}/verify")
    verified = bench.client.post(verify_url, headers=ADMIN, json={"note": "reproduced"})
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"
    only = bench.client.get(bench.url(f"/leaderboard/{SLUG}?verified_only=true")).json()
    assert [row["status"] for row in only] == ["verified"]

    again = bench.client.post(verify_url, headers=ADMIN, json={})
    assert again.status_code == 409
    assert again.json() == {"detail": "submission is verified"}

    withdrawn = bench.client.post(
        bench.url(f"/admin/submissions/{submission_id}/withdraw"),
        headers=ADMIN,
        json={},
    )
    assert withdrawn.json()["status"] == "withdrawn"
    assert bench.client.get(bench.url(f"/leaderboard/{SLUG}")).json() == []
    missing = bench.client.post(
        bench.url("/admin/submissions/nope/verify"), headers=ADMIN, json={}
    )
    assert missing.status_code == 404


def test_disabled_account_loses_access(bench: Bench) -> None:
    headers = bench.register()
    user_id = bench.client.get(bench.url("/auth/me"), headers=headers).json()["user_id"]
    disabled = bench.client.post(
        bench.url(f"/admin/users/{user_id}/disable"), headers=ADMIN
    )
    assert disabled.json() == {"message": f"user {user_id} updated"}
    assert bench.client.get(bench.url("/auth/me"), headers=headers).status_code == 401
    assert bench.sign_in() == {"login_error": "disabled"}
    assert (
        bench.client.post(
            bench.url("/admin/users/nope/disable"), headers=ADMIN
        ).status_code
        == 404
    )


def test_accounts_held_for_approval(
    tmp_path: Path, datasets_root: Path, labels: Labels, to_csv: Csv
) -> None:
    bench = Bench(tmp_path, datasets_root, require_approval=True)
    headers = bench.register()
    me = bench.client.get(bench.url("/auth/me"), headers=headers).json()
    assert me["approved"] is False
    held = bench.submit(headers, to_csv(labels))
    assert held.status_code == 403
    assert held.json() == {"detail": "this account awaits approval"}
    bench.client.post(bench.url(f"/admin/users/{me['user_id']}/approve"), headers=ADMIN)
    assert bench.submit(headers, to_csv(labels)).status_code == 201


def test_baselines_use_the_grader_and_skip_the_quota(
    bench: Bench, labels: Labels, to_csv: Csv
) -> None:
    raw = to_csv(labels)
    assert bench.submit({}, raw, path="/admin/baselines").status_code == 401
    knn = {**METADATA, "method_name": "kNN baseline", "model_family": "kNN"}
    linear = {**METADATA, "method_name": "linear baseline", "model_family": "linear"}
    first = bench.submit(ADMIN, raw, knn, path="/admin/baselines")
    second = bench.submit(ADMIN, raw, linear, path="/admin/baselines")
    assert (first.status_code, second.status_code) == (201, 201)  # no one-hour gap
    board = bench.client.get(bench.url(f"/leaderboard/{SLUG}")).json()
    assert sorted(row["method_name"] for row in board) == [
        "kNN baseline",
        "linear baseline",
    ]
    assert {row["is_baseline"] for row in board} == {True}
    assert {row["display_name"] for row in board} == {"TorchCell baselines"}
    with bench.sessions() as session:
        assert session.scalars(select(User.is_system)).all() == [True]
    # The system account has no sign-in identity, so no sign-in can reach it, and its
    # address is in the reserved ``.invalid`` domain, which is not a valid email claim.
    with bench.sessions() as session:
        system = session.scalars(select(User)).one()
        assert (system.oidc_issuer, system.oidc_subject) == (None, None)
    assert bench.sign_in(person("baselines@torchcell.invalid")) == {
        "login_error": "failed"
    }
    with bench.sessions() as session:
        assert len(session.scalars(select(User)).all()) == 1
    rejected = bench.submit(ADMIN, b"bad", knn, path="/admin/baselines")
    assert rejected.status_code == 422


# ------------------------------------------------------------------------- config


def _env(tmp_path: Path, datasets_root: Path) -> dict[str, str]:
    (tmp_path / "db_password").write_text("p@ss/word\n")
    (tmp_path / "jwt_secret").write_text("k" * 48 + "\n")
    (tmp_path / "cilogon_secret").write_text("cilogon-secret\n")
    (tmp_path / "idps.txt").write_text(f"{UIUC}\n\nhttps://orcid.org/oauth/authorize\n")
    (tmp_path / "admin_keys.json").write_text(json.dumps({"ops": "0" * 64}))
    (tmp_path / "blocked.txt").write_text("Mailinator.com\n\ntrashmail.com\n")
    return {
        "TC_BENCH_DB_HOST": "tc-bench-db",
        "TC_BENCH_DB_NAME": "tcbench",
        "TC_BENCH_DB_USER": "tcbench",
        "TC_BENCH_DB_PASSWORD_FILE": str(tmp_path / "db_password"),
        "TC_BENCH_DATASETS_ROOT": str(datasets_root),
        "TC_BENCH_SUBMISSIONS_ROOT": str(tmp_path),
        "TC_BENCH_JWT_SECRET_FILE": str(tmp_path / "jwt_secret"),
        "TC_BENCH_ADMIN_KEYS_FILE": str(tmp_path / "admin_keys.json"),
        "TC_BENCH_ACCOUNT_URL": ACCOUNT_URL,
        "TC_BENCH_PUBLIC_URL": "https://bench.example/bench/",
        "TC_BENCH_CILOGON_CLIENT_ID": CLIENT_ID,
        "TC_BENCH_CILOGON_CLIENT_SECRET_FILE": str(tmp_path / "cilogon_secret"),
        "TC_BENCH_ALLOWED_IDPS_FILE": str(tmp_path / "idps.txt"),
        "TC_BENCH_CORS_ORIGINS": "https://site.example, https://mjvolk3.github.io",
        "TC_BENCH_BLOCKED_DOMAINS_FILE": str(tmp_path / "blocked.txt"),
        "TC_BENCH_ALLOWED_DOMAIN_SUFFIXES": ".edu, .ac.uk",
        "TC_BENCH_TRUST_PROXY": "1",
    }


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name in list(__import__("os").environ):
        if name.startswith("TC_BENCH_"):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_config_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets_root: Path
) -> None:
    _set_env(monkeypatch, _env(tmp_path, datasets_root))
    config = BenchServerConfig.from_env()
    assert (
        config.database_url.get_secret_value()
        == "postgresql+psycopg://tcbench:p%40ss%2Fword@tc-bench-db:5432/tcbench"
    )
    assert config.jwt_secret.get_secret_value() == "k" * 48
    assert config.cors_origins == ("https://site.example", "https://mjvolk3.github.io")
    assert (config.host, config.port) == ("127.0.0.1", 8725)
    assert (config.trust_proxy, config.require_approval) == (True, False)
    assert (config.tier, config.build) == ("production", None)
    assert config.account_policy.blocked_domains == {"mailinator.com", "trashmail.com"}
    assert config.account_policy.allowed_domain_suffixes == (".edu", ".ac.uk")
    assert config.account_policy.allowed_idps == {
        UIUC,
        "https://orcid.org/oauth/authorize",
    }
    assert config.oidc.client_id == CLIENT_ID
    assert config.oidc.client_secret.get_secret_value() == "cilogon-secret"
    # A service published under a path prefix registers that prefix in its callback,
    # and the sign-in cookie is scoped to the same prefix.
    assert config.oidc.redirect_uri == (
        "https://bench.example/bench/api/v1/auth/callback"
    )
    assert config.oidc.cookie_path == "/bench/api/v1/auth"
    assert config.oidc.secure is True
    assert config.oidc.metadata_url == CILOGON_METADATA_URL
    assert config.admin_keys.hashes == {"ops": "0" * 64}
    for secret in ("p@ss", "kkkk", "cilogon-secret"):
        assert secret not in repr(config)


def test_config_optional_and_missing_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets_root: Path
) -> None:
    env = _env(tmp_path, datasets_root)
    del env["TC_BENCH_ALLOWED_IDPS_FILE"]
    env["TC_BENCH_OIDC_METADATA_URL"] = "https://test.cilogon.org/.well-known/x"
    _set_env(monkeypatch, env)
    config = BenchServerConfig.from_env()
    assert config.account_policy.allowed_idps == frozenset()
    assert config.oidc.metadata_url == "https://test.cilogon.org/.well-known/x"

    for required in (
        "TC_BENCH_JWT_SECRET_FILE",
        "TC_BENCH_PUBLIC_URL",
        "TC_BENCH_CILOGON_CLIENT_ID",
        "TC_BENCH_CILOGON_CLIENT_SECRET_FILE",
    ):
        _set_env(monkeypatch, {k: v for k, v in env.items() if k != required})
        with pytest.raises(KeyError, match=required):
            BenchServerConfig.from_env()


def test_config_refuses_a_short_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets_root: Path
) -> None:
    env = _env(tmp_path, datasets_root)
    (tmp_path / "jwt_secret").write_text("short")
    _set_env(monkeypatch, env)
    with pytest.raises(ValidationError, match="jwt secret must be at least 32 bytes"):
        BenchServerConfig.from_env()


def test_trust_proxy_reads_the_forwarded_address(
    tmp_path: Path, datasets_root: Path
) -> None:
    bench = Bench(tmp_path, datasets_root, trust_proxy=True, max_signups_per_address=1)

    def sign_in(email: str, forwarded: str) -> set[str]:
        return set(bench.sign_in(person(email), headers={"X-Forwarded-For": forwarded}))

    assert sign_in("a@example.org", "198.51.100.9, 203.0.113.7") == {"login_code"}
    # same proxy-appended address, different client-supplied prefix: limited
    assert sign_in("b@example.org", "10.0.0.1, 203.0.113.7") == {"login_error"}
    assert sign_in("c@example.org", "203.0.113.8") == {"login_code"}


# ---------------------------------------------------------------------------- cli


def test_main_gen_admin_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["tc-bench-server", "--gen-admin-key", "ops"])
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("server started"))
    app_module.main()
    out = capsys.readouterr().out
    assert "API key for 'ops'" in out
    assert "TC_BENCH_ADMIN_KEYS_FILE" in out


def test_main_runs_uvicorn_on_loopback_by_default(  # test-quality: allow main() reports only through the patched uvicorn.run
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets_root: Path
) -> None:
    _set_env(monkeypatch, _env(tmp_path, datasets_root))
    engine = create_engine(f"sqlite:///{tmp_path / 'main.db'}")
    monkeypatch.setattr(app_module, "make_engine", lambda url: engine)
    started: dict[str, Any] = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: started.update(host=host, port=port)
    )

    monkeypatch.setattr(sys, "argv", ["tc-bench-server", "--init-db"])
    app_module.main()
    assert started == {}
    from sqlalchemy import inspect

    assert sorted(inspect(engine).get_table_names()) == [
        "api_tokens",
        "login_codes",
        "submissions",
        "users",
    ]

    monkeypatch.setattr(sys, "argv", ["tc-bench-server"])
    app_module.main()
    assert started == {"host": "127.0.0.1", "port": 8725}
    monkeypatch.setattr(sys, "argv", ["tc-bench-server", "--port", "9000"])
    app_module.main()
    assert started == {"host": "127.0.0.1", "port": 9000}


# -------------------------------------------------------------------- binary task


def test_binary_dataset_is_scored_with_ranking_metrics(tmp_path: Path) -> None:
    from torchcell.benchmark.bundle import write_bundle
    from torchcell.benchmark.submission import Split

    datasets_root = tmp_path / "datasets"
    datasets_root.mkdir()
    genes = {
        Split.TRAIN: ["g0"],
        Split.VAL: ["g1", "g2", "g3", "g4"],
        Split.TEST: ["g5", "g6", "g7", "g8"],
    }
    essential = {"g1", "g3", "g5", "g6"}
    write_bundle(
        datasets_root,
        slug="toy-essential",
        title="Toy essentiality",
        description="Eight genes with a 0/1 label.",
        loader_class="ToyEssentialityDataset",
        citation_key="toy2026",
        version="1",
        task="binary",
        primary_metric="auroc",
        splits=genes,
        values={
            (gene, "is_essential"): float(gene in essential)
            for split in (Split.VAL, Split.TEST)
            for gene in genes[split]
        },
    )
    bench = Bench(tmp_path, datasets_root)
    headers = bench.register()
    assert bench.client.get(bench.url("/datasets")).json()[0]["task"] == "binary"

    # Validation ranks 1, 0, 1, 0 (AUROC 3/4, AUPRC 5/6); test ranks both essential
    # genes first.
    scores = {"g1": 0.9, "g2": 0.8, "g3": 0.7, "g4": 0.1}
    scores |= {"g5": 5.0, "g6": 4.0, "g7": -1.0, "g8": -2.0}
    lines = ["record_id,split,target,prediction"]
    for gene, value in scores.items():
        split = "val" if gene in genes[Split.VAL] else "test"
        lines.append(f"{gene},{split},is_essential,{value}")
    response = bench.client.post(
        bench.url("/submissions"),
        headers=headers,
        data={"dataset": "toy-essential", "metadata": json.dumps(METADATA)},
        files={"predictions": ("p.csv", "\n".join(lines).encode(), "text/csv")},
    )
    assert response.status_code == 201
    result = response.json()
    assert result["val"]["macro"] == {
        "auroc": pytest.approx(0.75),
        "auprc": pytest.approx(5 / 6),
    }
    assert result["test"]["macro"] == {"auroc": 1.0, "auprc": 1.0}
    assert result["flags"] == ["test_exceeds_val"]  # 1.0 against 0.75 on the primary
    board = bench.client.get(bench.url("/leaderboard/toy-essential")).json()
    assert board[0]["test"]["per_target"] == {
        "is_essential": {"auroc": 1.0, "auprc": 1.0}
    }


def test_tier_and_build_come_from_the_environment_and_show_in_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets_root: Path
) -> None:
    env = _env(tmp_path, datasets_root)
    _set_env(
        monkeypatch,
        {**env, "TC_BENCH_TIER": "staging", "TC_BENCH_BUILD_COMMIT": "abc1234"},
    )
    config = BenchServerConfig.from_env()
    assert (config.tier, config.build) == ("staging", "abc1234")

    bench = Bench(tmp_path, datasets_root, tier="staging", build="abc1234")
    assert bench.client.get(bench.url("/health")).json() == {
        "status": "ok",
        "n_datasets": 1,
        "tier": "staging",
        "build": "abc1234",
    }

    _set_env(monkeypatch, {**env, "TC_BENCH_TIER": "preview"})
    with pytest.raises(ValidationError, match="tier"):
        BenchServerConfig.from_env()
