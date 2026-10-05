# torchcell/benchmark/app.py
# [[torchcell.benchmark.app]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/app.py
# Test file: tests/torchcell/benchmark/test_app.py

"""``tc-bench``: the HTTP service behind the public benchmark leaderboard.

Reading the board needs no account. Submitting needs an account, a bearer credential,
and quota (see :mod:`torchcell.benchmark.ratelimit`). There are no passwords: a person
signs in at CILogon with their institution, and the service keeps the identity CILogon
asserts (see :mod:`torchcell.benchmark.oidc`). Sign-in is three steps:

1. The account page sends the browser to ``GET /auth/login``, which redirects to
   CILogon with a ``state``, a ``nonce`` and a PKCE challenge. Those three are kept in
   a signed, short-lived cookie scoped to the sign-in routes.
2. CILogon redirects back to ``GET /auth/callback``. The response is verified, the
   account is found or created, and the browser is sent to the account page with a
   one-time sign-in code in the URL fragment (or an error code when refused).
3. The account page posts the code to ``POST /auth/exchange`` and receives the bearer
   token. The code works once and expires within minutes, so the address bar and the
   browser history only ever hold a spent code, and the token never appears in a URL.

The bearer credential is one of two things. A browser holds the session token of step
3. A script holds a personal API token (``tcb_...``), which the account's owner creates
on the account page (``POST /auth/tokens``, shown once, stored as its sha256) and can
revoke there. A token expires after the lifetime chosen when it is created
(``api_token_default_days`` unless the request names one, never more than
``api_token_max_days``); a revoked or expired token's row is deleted, so the table
holds at most ``max_api_tokens`` rows per account. A new token replaces an old one;
nothing is renewed in place. Both are sent as ``Authorization: Bearer``. An API token reads the
account, reads the quota, submits and lists the account's attempts; changing the
profile and managing tokens need the session token, so a leaked API token cannot mint
others. The quota is per account, whichever credential an attempt arrives with. The
website's form and :mod:`torchcell.benchmark.client` call the same
``POST /submissions``.

A submission is a multipart upload of a predictions CSV and a metadata JSON; the
server validates both with the pydantic models of
:mod:`torchcell.benchmark.submission`, rejects a malformed upload with explicit
reasons (HTTP 422), and otherwise grades it, flags it, archives it as a zip and returns
the scores with the status ``provisional``. An admin (``X-API-Key``, the same named-key
scheme as ``tc-data``) promotes a reproduced submission to ``verified``, withdraws one,
approves or disables an account, and uploads baselines through the same grader.

Every attempt is one ``submissions`` row, rejected ones included, because the quota
counts attempts. The quota check and the insert run in one transaction under a lock on
the user's row, so concurrent uploads cannot both pass.

The server binds 127.0.0.1 by default and is meant to sit behind a TLS reverse proxy;
it is never the process that listens on a public port. Configuration is all
environment-driven (``TC_BENCH_*``), with every secret read from a file:

- ``TC_BENCH_DB_HOST``, ``TC_BENCH_DB_PORT`` (5432), ``TC_BENCH_DB_NAME``,
  ``TC_BENCH_DB_USER``, ``TC_BENCH_DB_PASSWORD_FILE``: the PostgreSQL database.
- ``TC_BENCH_DATASETS_ROOT``: the bundles (:mod:`torchcell.benchmark.bundle`), loaded
  once at startup, so adding a dataset needs a restart.
- ``TC_BENCH_SUBMISSIONS_ROOT``: where scored submissions are archived.
- ``TC_BENCH_JWT_SECRET_FILE``: the session signing secret (32 bytes or more).
- ``TC_BENCH_ADMIN_KEYS_FILE``: JSON ``{name: sha256hex}`` of the admin keys.
- ``TC_BENCH_ACCOUNT_URL``: public URL of the site's account page; sign-in ends there.
- ``TC_BENCH_PUBLIC_URL``: public base URL of this service as the browser reaches it,
  without the ``/api/v1`` prefix. The callback registered with CILogon is this URL
  plus ``/api/v1/auth/callback`` and must match the registration exactly.
- ``TC_BENCH_CILOGON_CLIENT_ID``, ``TC_BENCH_CILOGON_CLIENT_SECRET_FILE``: the client
  registered at https://cilogon.org/oauth2/register.
- ``TC_BENCH_CORS_ORIGINS``: comma-separated origins of the website.
- Optional: ``TC_BENCH_HOST`` (127.0.0.1), ``TC_BENCH_PORT`` (8725),
  ``TC_BENCH_TRUST_PROXY`` (``1`` when behind the reverse proxy, so the client address
  is read from ``X-Forwarded-For``), ``TC_BENCH_REQUIRE_APPROVAL`` (``1`` to hold new
  accounts until an admin approves them), ``TC_BENCH_ALLOWED_IDPS_FILE`` (one identity
  provider entity id per line; unset accepts every provider CILogon offers),
  ``TC_BENCH_BLOCKED_DOMAINS_FILE`` (one email domain per line),
  ``TC_BENCH_ALLOWED_DOMAIN_SUFFIXES`` (comma-separated),
  ``TC_BENCH_OIDC_METADATA_URL`` (CILogon's discovery document by default),
  ``TC_BENCH_TIER`` (``production`` by default; ``staging`` or ``development``
  otherwise) and ``TC_BENCH_BUILD_COMMIT`` (the commit the image was built from, set
  by the image). ``/health`` reports both, which is how a deploy is checked.

Staging and production are two deployments of this one service from one branch: each
has its own database, archive directory, secrets, port and site, and nothing is shared
between them. See ``docker-compose.tc-bench.yml`` and ``scripts/tc_bench_deploy.sh``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlencode

import httpx2
import uvicorn
from authlib.integrations.base_client import OAuthError
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from joserfc.errors import JoseError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
    model_validator,
)
from sqlalchemy import URL, Engine, delete, func, select
from sqlalchemy.orm import Session
from starlette.datastructures import Headers
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from torchcell.api_keys import API_KEY_HEADER, ApiKeys, print_minted_key
from torchcell.benchmark.bundle import (
    SPLITS_FILENAME,
    TEMPLATE_FILENAME,
    BenchmarkBundle,
    BenchmarkDatasetPublic,
    load_bundles,
    sha256_bytes,
)
from torchcell.benchmark.db import (
    BOARD_STATUSES,
    ApiToken,
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
from torchcell.benchmark.grading import SplitScores, score
from torchcell.benchmark.integrity import IntegrityPolicy, flag_submission, oriented
from torchcell.benchmark.oidc import (
    CILOGON_METADATA_URL,
    DISPLAY_NAME_MAX_LENGTH,
    DISPLAY_NAME_MIN_LENGTH,
    LOGIN_STATE_COOKIE,
    LOGIN_STATE_MAX_AGE_SECONDS,
    Identity,
    LoginError,
    LoginRefused,
    OidcConfig,
    build_provider,
    callback_url,
    identity_from_claims,
)
from torchcell.benchmark.ratelimit import QuotaStatus, SubmissionLimits, evaluate_quota
from torchcell.benchmark.results import Quota, SubmissionResult
from torchcell.benchmark.security import (
    JWT_SECRET_MIN_BYTES,
    AccountPolicy,
    canonical_email,
    decode_access_token,
    derive_key,
    hash_client_address,
    hash_token,
    is_api_token,
    issue_access_token,
    new_api_token,
    new_one_time_token,
)
from torchcell.benchmark.storage import archive_submission
from torchcell.benchmark.submission import (
    Split,
    SubmissionMetadata,
    submission_json_schema,
)
from torchcell.benchmark.validation import validate_predictions

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

APP_TITLE = "torchcell benchmark endpoint"
APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8725
ADMIN_KEYS_FILE_VAR = "TC_BENCH_ADMIN_KEYS_FILE"
MAX_METADATA_BYTES = 16 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
SIGNUP_WINDOW = timedelta(hours=24)
BASELINE_USER_EMAIL = "baselines@torchcell.invalid"
BASELINE_USER_NAME = "TorchCell baselines"
LOGIN_CODE_FRAGMENT = "login_code"
LOGIN_ERROR_FRAGMENT = "login_error"


class BenchServerConfig(BaseModel):
    """Runtime configuration of the benchmark service."""

    model_config = ConfigDict(frozen=True)

    database_url: SecretStr
    datasets_root: Path
    submissions_root: Path
    jwt_secret: SecretStr
    admin_keys: ApiKeys
    account_url: str = Field(description="Public URL of the website's account page.")
    cors_origins: tuple[str, ...]
    oidc: OidcConfig
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    trust_proxy: bool = False
    require_approval: bool = False
    max_upload_bytes: int = 16 * 1024 * 1024
    limits: SubmissionLimits = SubmissionLimits()
    integrity: IntegrityPolicy = IntegrityPolicy()
    account_policy: AccountPolicy = AccountPolicy()
    access_token_ttl: timedelta = timedelta(hours=12)
    login_code_ttl: timedelta = timedelta(minutes=2)
    max_signups_per_address: int = 5
    tier: Tier = "production"
    build: str | None = None
    max_api_tokens: int = 5
    api_token_default_days: int = 90
    api_token_max_days: int = 365

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.jwt_secret.get_secret_value().encode()) < JWT_SECRET_MIN_BYTES:
            raise ValueError(
                f"jwt secret must be at least {JWT_SECRET_MIN_BYTES} bytes"
            )
        return self

    @classmethod
    def from_env(cls) -> Self:
        """Build the config from ``TC_BENCH_*``; a missing required variable raises."""
        env = os.environ

        def secret_file(name: str) -> str:
            return Path(env[name]).read_text(encoding="utf-8").strip()

        database_url = URL.create(
            "postgresql+psycopg",
            username=env["TC_BENCH_DB_USER"],
            password=secret_file("TC_BENCH_DB_PASSWORD_FILE"),
            host=env["TC_BENCH_DB_HOST"],
            port=int(env.get("TC_BENCH_DB_PORT", "5432")),
            database=env["TC_BENCH_DB_NAME"],
        ).render_as_string(hide_password=False)
        idps_file = env.get("TC_BENCH_ALLOWED_IDPS_FILE")
        allowed_idps = (
            frozenset(
                line.strip()
                for line in Path(idps_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if idps_file
            else frozenset()
        )
        blocked_file = env.get("TC_BENCH_BLOCKED_DOMAINS_FILE")
        blocked = (
            frozenset(
                line.strip().lower()
                for line in Path(blocked_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if blocked_file
            else frozenset()
        )
        suffixes = tuple(
            s.strip().lower()
            for s in env.get("TC_BENCH_ALLOWED_DOMAIN_SUFFIXES", "").split(",")
            if s.strip()
        )
        return cls.model_validate(
            {
                "database_url": database_url,
                "datasets_root": env["TC_BENCH_DATASETS_ROOT"],
                "submissions_root": env["TC_BENCH_SUBMISSIONS_ROOT"],
                "jwt_secret": secret_file("TC_BENCH_JWT_SECRET_FILE"),
                "admin_keys": ApiKeys.from_file(env[ADMIN_KEYS_FILE_VAR]),
                "account_url": env["TC_BENCH_ACCOUNT_URL"],
                "cors_origins": tuple(
                    o.strip()
                    for o in env["TC_BENCH_CORS_ORIGINS"].split(",")
                    if o.strip()
                ),
                "oidc": OidcConfig(
                    client_id=env["TC_BENCH_CILOGON_CLIENT_ID"],
                    client_secret=SecretStr(
                        secret_file("TC_BENCH_CILOGON_CLIENT_SECRET_FILE")
                    ),
                    redirect_uri=callback_url(env["TC_BENCH_PUBLIC_URL"], API_PREFIX),
                    metadata_url=env.get(
                        "TC_BENCH_OIDC_METADATA_URL", CILOGON_METADATA_URL
                    ),
                ),
                "host": env.get("TC_BENCH_HOST", DEFAULT_HOST),
                "port": int(env.get("TC_BENCH_PORT", str(DEFAULT_PORT))),
                "trust_proxy": env.get("TC_BENCH_TRUST_PROXY", "0") == "1",
                "require_approval": env.get("TC_BENCH_REQUIRE_APPROVAL", "0") == "1",
                "tier": env.get("TC_BENCH_TIER", "production"),
                "build": env.get("TC_BENCH_BUILD_COMMIT") or None,
                "account_policy": AccountPolicy(
                    blocked_domains=blocked,
                    allowed_domain_suffixes=suffixes,
                    allowed_idps=allowed_idps,
                ),
            }
        )


# ---------------------------------------------------------------- request and response

DisplayName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=DISPLAY_NAME_MIN_LENGTH,
        max_length=DISPLAY_NAME_MAX_LENGTH,
    ),
]
Affiliation = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]


class ExchangeRequest(BaseModel):
    """Body of ``POST /auth/exchange``."""

    model_config = ConfigDict(extra="forbid")

    code: Annotated[str, StringConstraints(min_length=1, max_length=256)]


class ProfileRequest(BaseModel):
    """Body of ``POST /auth/profile``: what the board shows about the account."""

    model_config = ConfigDict(extra="forbid")

    display_name: DisplayName
    affiliation: Affiliation | None = None


class ApiTokenRequest(BaseModel):
    """Body of ``POST /auth/tokens``."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)
    ] = Field(description="A label for the token, for example the machine it is on.")
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        description="Days until the token expires; the server's default when omitted.",
    )


class ReviewRequest(BaseModel):
    """Body of the admin verify and withdraw routes."""

    note: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""


class Message(BaseModel):
    """A plain acknowledgment."""

    message: str


Tier = Literal["development", "staging", "production"]


class Health(BaseModel):
    """Liveness summary (no auth)."""

    status: str
    n_datasets: int
    tier: Tier = Field(description="Which deployment answered.")
    build: str | None = Field(
        description="The git commit the running image was built from, when recorded."
    )


class TokenResponse(BaseModel):
    """A session token."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime


class UserPublic(BaseModel):
    """What the board shows about an account."""

    user_id: str
    display_name: str
    affiliation: str | None
    identity_provider: str | None = Field(
        description="Name of the provider the account signs in through."
    )
    created_at: datetime


class MeResponse(UserPublic):
    """The signed-in account as its owner sees it."""

    email: str
    approved: bool


class ApiTokenInfo(BaseModel):
    """A personal API token as its owner sees it after creation (never the token)."""

    token_id: str
    name: str
    hint: str = Field(description="The first characters of the token.")
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime | None


class ApiTokenCreated(ApiTokenInfo):
    """A new personal API token; ``token`` is returned this once and not stored."""

    token: str


class LeaderboardRow(BaseModel):
    """One scored submission on a dataset's board."""

    submission_id: str
    user_id: str
    display_name: str
    affiliation: str | None
    method_name: str
    model_family: str
    encoding: str
    code_url: str | None
    status: SubmissionStatus
    submitted_at: datetime
    is_baseline: bool
    val: SplitScores
    test: SplitScores
    flags: list[str]


class UserHistoryRow(LeaderboardRow):
    """A leaderboard row that also names its dataset."""

    dataset_slug: str


class UserHistory(BaseModel):
    """An account's public record: every scored submission, oldest first."""

    user: UserPublic
    submissions: list[UserHistoryRow]


def _user_public(user: User) -> UserPublic:
    return UserPublic(
        user_id=user.id,
        display_name=user.display_name,
        affiliation=user.affiliation,
        identity_provider=user.idp_name,
        created_at=user.created_at,
    )


def _scores(raw: dict[str, Any] | None) -> SplitScores | None:
    return None if raw is None else SplitScores.model_validate(raw)


def _result(submission: Submission) -> SubmissionResult:
    return SubmissionResult(
        submission_id=submission.id,
        dataset_slug=submission.dataset_slug,
        status=SubmissionStatus(submission.status),
        submitted_at=submission.created_at,
        method_name=submission.method_name,
        rejection_reasons=submission.rejection_reasons,
        val=_scores(submission.val_scores),
        test=_scores(submission.test_scores),
        flags=submission.flags,
        archive_sha256=submission.archive_sha256,
    )


def _row(submission: Submission, user: User) -> UserHistoryRow:
    return UserHistoryRow(
        submission_id=submission.id,
        user_id=user.id,
        display_name=user.display_name,
        affiliation=user.affiliation,
        method_name=submission.method_name,
        model_family=submission.model_family or "",
        encoding=submission.encoding or "",
        code_url=submission.code_url,
        status=SubmissionStatus(submission.status),
        submitted_at=submission.created_at,
        is_baseline=submission.is_baseline,
        val=SplitScores.model_validate(submission.val_scores),
        test=SplitScores.model_validate(submission.test_scores),
        flags=submission.flags,
        dataset_slug=submission.dataset_slug,
    )


# ------------------------------------------------------------------------- middleware


class UploadSizeGuard:
    """Refuse an oversized upload from its ``Content-Length``, before the body is read.

    The framework spools a multipart body to disk before the route runs, so the size
    limit has to be applied here. A request without ``Content-Length`` is refused (411)
    because its size cannot be bounded up front.
    """

    def __init__(self, app: ASGIApp, *, paths: frozenset[str], max_bytes: int) -> None:
        """Guard ``POST`` requests to ``paths`` with a body limit of ``max_bytes``."""
        self.app = app
        self.paths = paths
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Answer 411 or 413 for a guarded upload that is unbounded or too large."""
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"] in self.paths
        ):
            length = Headers(scope=scope).get("content-length")
            if length is None or not length.isdigit():
                response = JSONResponse(
                    {"detail": "Content-Length is required for uploads"},
                    status_code=411,
                )
                await response(scope, receive, send)
                return
            if int(length) > self.max_bytes:
                response = JSONResponse(
                    {"detail": f"upload exceeds {self.max_bytes} bytes"},
                    status_code=413,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ------------------------------------------------------------------------------- app


def create_app(
    config: BenchServerConfig,
    *,
    engine: Engine | None = None,
    oidc_transport: Any = None,
    clock: Callable[[], datetime] = utcnow,
) -> FastAPI:
    """Build the FastAPI app bound to ``config``.

    ``engine`` and ``clock`` default to what ``config`` describes and the wall clock.
    ``oidc_transport`` replaces the HTTP transport of the calls to the identity
    provider; tests pass one that answers as CILogon.
    """
    db_engine = engine or make_engine(config.database_url.get_secret_value())
    sessions = make_session_factory(db_engine)
    provider = build_provider(config.oidc, oidc_transport)
    bundles = load_bundles(config.datasets_root)
    secret = config.jwt_secret.get_secret_value()

    app = FastAPI(
        title=APP_TITLE,
        summary="Accounts, submissions and leaderboards of the TorchCell benchmark.",
        description=(
            "Reading datasets and leaderboards needs no account. Submitting needs an "
            "account (sign in through CILogon) and a bearer credential: the session "
            "token of a browser, or a personal API token created on the account page. "
            "Submit predictions, not scores: the server validates the upload against the dataset's template, grades it on "
            "the validation and test splits, and returns the result."
        ),
        version=APP_VERSION,
    )
    app.state.config = config
    app.state.engine = db_engine
    app.add_middleware(
        UploadSizeGuard,
        paths=frozenset({f"{API_PREFIX}/submissions", f"{API_PREFIX}/admin/baselines"}),
        max_bytes=config.max_upload_bytes
        + MAX_METADATA_BYTES
        + MULTIPART_OVERHEAD_BYTES,
    )
    # Holds the state, nonce and PKCE verifier of a sign-in in progress, signed, for
    # the few minutes between the redirect to CILogon and the callback. It is sent only
    # to the sign-in routes and is not the session: that is the bearer token.
    app.add_middleware(
        SessionMiddleware,
        secret_key=derive_key(secret, "login-state"),
        session_cookie=LOGIN_STATE_COOKIE,
        max_age=LOGIN_STATE_MAX_AGE_SECONDS,
        path=config.oidc.cookie_path,
        same_site="lax",
        https_only=config.oidc.secure,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    bearer_scheme = HTTPBearer(auto_error=False)
    admin_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

    def get_session() -> Iterator[Session]:
        with sessions() as session:
            yield session

    def unauthorized(detail: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

    def active(user: User | None, detail: str) -> User:
        if user is None or user.disabled:
            raise unauthorized(detail)
        return user

    def session_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
        session: Session = Depends(get_session),
    ) -> User:
        """The account behind a session token; a personal API token is not accepted."""
        user_id = (
            decode_access_token(credentials.credentials, secret)
            if credentials
            else None
        )
        return active(
            session.get(User, user_id) if user_id else None,
            "sign in on the account page to use this route",
        )

    def current_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
        session: Session = Depends(get_session),
    ) -> User:
        """The account behind a session token or a personal API token."""
        if credentials is None or not is_api_token(credentials.credentials):
            return session_user(credentials, session)
        row = session.scalar(
            select(ApiToken).where(
                ApiToken.token_sha256 == hash_token(credentials.credentials)
            )
        )
        now = clock()
        refusal = "invalid, revoked or expired API token"
        if row is None or row.expires_at <= now:
            raise unauthorized(refusal)
        user = active(session.get(User, row.user_id), refusal)
        row.last_used_at = now
        session.commit()
        return user

    def require_admin(api_key: str | None = Depends(admin_scheme)) -> str:
        name = config.admin_keys.verify(api_key) if api_key else None
        if name is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing admin key",
            )
        return name

    def client_address(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if config.trust_proxy and forwarded:
            # The last entry is the one our own proxy appended.
            return forwarded.split(",")[-1].strip()
        return request.client.host if request.client else "unknown"

    def get_bundle(slug: str) -> BenchmarkBundle:
        bundle = bundles.get(slug)
        if bundle is None:
            raise HTTPException(status_code=404, detail="unknown benchmark dataset")
        return bundle

    def quota_state(session: Session, user_id: str, now: datetime) -> QuotaStatus:
        attempts = session.scalars(
            select(Submission.created_at).where(
                Submission.user_id == user_id,
                Submission.is_baseline.is_(False),
                Submission.created_at > now - config.limits.window,
            )
        ).all()
        return evaluate_quota(attempts, now, config.limits)

    def to_quota(state: QuotaStatus) -> Quota:
        return Quota(
            max_per_window=config.limits.max_per_window,
            window_hours=config.limits.window.total_seconds() / 3600,
            min_gap_minutes=config.limits.min_gap.total_seconds() / 60,
            used_in_window=state.used_in_window,
            remaining=state.remaining if state.allowed else 0,
            next_allowed_at=state.next_allowed_at,
        )

    def process_submission(
        session: Session,
        user: User,
        bundle: BenchmarkBundle,
        metadata_raw: str,
        raw: bytes,
        now: datetime,
        *,
        is_baseline: bool,
    ) -> Submission:
        """Record one attempt: validate, then grade, flag and archive when it is valid."""
        submission = Submission(
            id=new_id(),
            user_id=user.id,
            dataset_slug=bundle.dataset.slug,
            dataset_version=bundle.dataset.version,
            status=SubmissionStatus.REJECTED,
            is_baseline=is_baseline,
            created_at=now,
            method_name="(metadata not parsed)",
            upload_sha256=sha256_bytes(raw),
            rejection_reasons=[],
            flags=[],
        )
        session.add(submission)

        reasons: list[str] = []
        metadata: SubmissionMetadata | None = None
        if len(metadata_raw.encode("utf-8")) > MAX_METADATA_BYTES:
            reasons.append(f"metadata exceeds {MAX_METADATA_BYTES} bytes")
        else:
            try:
                metadata = SubmissionMetadata.model_validate_json(metadata_raw)
            except ValidationError as error:
                reasons.extend(
                    f"metadata.{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in error.errors()[:10]
                )
        if metadata is not None:
            submission.method_name = metadata.method_name
            submission.model_family = metadata.model_family
            submission.encoding = metadata.encoding
            submission.code_url = str(metadata.code_url) if metadata.code_url else None
            submission.submission_metadata = metadata.model_dump(mode="json")

        predictions: dict[tuple[str, str], float] | None = None
        if len(raw) > config.max_upload_bytes:
            reasons.append(f"predictions file exceeds {config.max_upload_bytes} bytes")
        else:
            report, predictions = validate_predictions(raw, bundle.spec)
            submission.n_rows = report.n_rows
            reasons.extend(report.reasons)

        if reasons or metadata is None or predictions is None:
            submission.rejection_reasons = reasons
            return submission

        scores = score(
            predictions, bundle.labels, bundle.spec.expected, bundle.dataset.task
        )
        val, test = scores[Split.VAL], scores[Split.TEST]
        primary = bundle.dataset.primary_metric
        earlier = session.scalars(
            select(Submission)
            .where(
                Submission.user_id == user.id,
                Submission.dataset_slug == bundle.dataset.slug,
                Submission.status.in_(BOARD_STATUSES),
                Submission.id != submission.id,
            )
            .order_by(Submission.created_at)
        ).all()
        history = [
            (
                getattr(SplitScores.model_validate(s.val_scores).macro, primary),
                getattr(SplitScores.model_validate(s.test_scores).macro, primary),
            )
            for s in earlier
        ]
        flags = flag_submission(
            primary,
            getattr(val.macro, primary),
            getattr(test.macro, primary),
            history,
            config.integrity,
        )
        submission.status = SubmissionStatus.PROVISIONAL
        submission.val_scores = val.model_dump(mode="json")
        submission.test_scores = test.model_dump(mode="json")
        submission.flags = [flag.value for flag in flags]
        record = archive_submission(
            config.submissions_root,
            bundle.dataset.slug,
            submission.id,
            now,
            {
                "predictions.csv": raw,
                "metadata.json": metadata.model_dump_json(indent=2).encode("utf-8"),
                "result.json": _result(submission)
                .model_dump_json(indent=2)
                .encode("utf-8"),
            },
        )
        submission.archive_path = record.relative_path
        submission.archive_sha256 = record.sha256
        return submission

    def respond(submission: Submission) -> SubmissionResult:
        result = _result(submission)
        if result.status == SubmissionStatus.REJECTED:
            raise HTTPException(status_code=422, detail=result.model_dump(mode="json"))
        return result

    router = APIRouter(prefix=API_PREFIX)

    # ---------------------------------------------------------------------- service

    @router.get("/health", response_model=Health, tags=["service"])
    def health() -> Health:
        """Liveness and the number of benchmark datasets loaded. No account needed."""
        return Health(
            status="ok", n_datasets=len(bundles), tier=config.tier, build=config.build
        )

    @router.get("/submission-schema", tags=["service"])
    def submission_schema() -> dict[str, Any]:
        """JSON Schema of a prediction row and of the metadata, plus the CSV columns."""
        return submission_json_schema()

    # ------------------------------------------------------------------------- auth

    def back(fragment: str, value: str) -> RedirectResponse:
        """Send the browser to the account page with one value in the URL fragment."""
        return RedirectResponse(
            f"{config.account_url}#{urlencode({fragment: value})}", status_code=303
        )

    def register(
        session: Session, identity: Identity, address: str, now: datetime
    ) -> User:
        """Create the account of a first sign-in, or refuse it with a reason."""
        if config.account_policy.rejection_reason(identity.email) is not None:
            raise LoginRefused(LoginError.EMAIL_NOT_ALLOWED)
        address_hash = hash_client_address(address, secret)
        recent = session.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.signup_address_hash == address_hash,
                User.created_at > now - SIGNUP_WINDOW,
            )
        )
        if (recent or 0) >= config.max_signups_per_address:
            raise LoginRefused(LoginError.TOO_MANY_ACCOUNTS)
        canonical = canonical_email(identity.email)
        # One account per address family: the same address arriving through a second
        # identity provider is refused, not merged, so it can neither open a second
        # account nor take over the first.
        if session.scalar(select(User.id).where(User.email_canonical == canonical)):
            raise LoginRefused(LoginError.EMAIL_IN_USE)
        user = User(
            id=new_id(),
            oidc_issuer=identity.issuer,
            oidc_subject=identity.subject,
            idp=identity.idp,
            idp_name=identity.idp_name,
            email=identity.email.lower(),
            email_canonical=canonical,
            display_name=identity.display_name,
            approved=not config.require_approval,
            signup_address_hash=address_hash,
            created_at=now,
        )
        session.add(user)
        session.flush()
        return user

    def complete_login(claims: dict[str, Any], address: str) -> str:
        """Find or create the account behind verified ``claims``; return a sign-in code."""
        now = clock()
        identity = identity_from_claims(claims)
        if not config.account_policy.idp_allowed(identity.idp):
            raise LoginRefused(LoginError.IDP_NOT_ALLOWED)
        with sessions() as session:
            user = session.scalar(
                select(User).where(
                    User.oidc_issuer == identity.issuer,
                    User.oidc_subject == identity.subject,
                )
            )
            if user is None:
                user = register(session, identity, address, now)
            if user.disabled:
                raise LoginRefused(LoginError.DISABLED)
            user.last_login_at = now
            session.execute(
                delete(LoginCode).where(
                    LoginCode.user_id == user.id, LoginCode.expires_at <= now
                )
            )
            code, code_hash = new_one_time_token()
            session.add(
                LoginCode(
                    id=new_id(),
                    user_id=user.id,
                    code_sha256=code_hash,
                    expires_at=now + config.login_code_ttl,
                    created_at=now,
                )
            )
            session.commit()
        return code

    @router.get("/auth/login", tags=["auth"], status_code=302)
    async def login(request: Request) -> RedirectResponse:
        """Start a sign-in: redirect the browser to CILogon."""
        try:
            redirect: RedirectResponse = await provider.authorize_redirect(
                request, config.oidc.redirect_uri
            )
        except httpx2.HTTPError:
            return back(LOGIN_ERROR_FRAGMENT, LoginError.UNAVAILABLE)
        return redirect

    @router.get("/auth/callback", tags=["auth"], status_code=303)
    async def callback(request: Request) -> RedirectResponse:
        """Finish a sign-in: verify CILogon's response and return to the account page.

        The account page receives ``#login_code=<one-time code>`` on success and
        ``#login_error=<reason>`` otherwise (see
        :class:`torchcell.benchmark.oidc.LoginError`).
        """
        try:
            token = await provider.authorize_access_token(request)
        except OAuthError as error:
            log.info("sign-in refused by the provider or the state check: %s", error)
            refused = error.error == "access_denied"
            return back(
                LOGIN_ERROR_FRAGMENT,
                LoginError.DENIED if refused else LoginError.FAILED,
            )
        except JoseError as error:
            log.warning("sign-in ID token did not verify: %s", error)
            return back(LOGIN_ERROR_FRAGMENT, LoginError.FAILED)
        except httpx2.HTTPError as error:
            log.warning("identity provider unreachable: %s", error)
            return back(LOGIN_ERROR_FRAGMENT, LoginError.UNAVAILABLE)
        claims = token.get("userinfo")
        if claims is None:
            return back(LOGIN_ERROR_FRAGMENT, LoginError.FAILED)
        try:
            code = await run_in_threadpool(
                complete_login, dict(claims), client_address(request)
            )
        except LoginRefused as refusal:
            return back(LOGIN_ERROR_FRAGMENT, refusal.error)
        return back(LOGIN_CODE_FRAGMENT, code)

    @router.post("/auth/exchange", response_model=TokenResponse, tags=["auth"])
    def exchange(
        body: ExchangeRequest, session: Session = Depends(get_session)
    ) -> TokenResponse:
        """Trade the one-time sign-in code from the callback for a bearer token."""
        now = clock()
        row = session.scalar(
            select(LoginCode)
            .where(LoginCode.code_sha256 == hash_token(body.code))
            .with_for_update()
        )
        if row is None or row.used_at is not None or row.expires_at <= now:
            raise HTTPException(status_code=400, detail="invalid or expired code")
        row.used_at = now
        user = session.get_one(User, row.user_id)
        session.commit()
        if user.disabled:
            raise HTTPException(status_code=403, detail="this account is disabled")
        # Wall clock, not ``clock``: the token library checks ``iat`` and ``exp`` against
        # the wall clock when it decodes, so a token must be issued on the same one.
        token, expires_at = issue_access_token(
            user.id, secret, config.access_token_ttl, utcnow()
        )
        return TokenResponse(access_token=token, expires_at=expires_at)

    @router.get("/auth/me", response_model=MeResponse, tags=["auth"])
    def me(user: User = Depends(current_user)) -> MeResponse:
        """The account the credential belongs to."""
        return MeResponse(
            **_user_public(user).model_dump(), email=user.email, approved=user.approved
        )

    @router.post("/auth/profile", response_model=MeResponse, tags=["auth"])
    def update_profile(
        body: ProfileRequest,
        user: User = Depends(session_user),
        session: Session = Depends(get_session),
    ) -> MeResponse:
        """Set the display name and affiliation the board shows for the account."""
        user.display_name = body.display_name
        user.affiliation = body.affiliation
        session.commit()
        return MeResponse(
            **_user_public(user).model_dump(), email=user.email, approved=user.approved
        )

    def token_info(row: ApiToken) -> ApiTokenInfo:
        return ApiTokenInfo(
            token_id=row.id,
            name=row.name,
            hint=row.token_hint,
            created_at=row.created_at,
            expires_at=row.expires_at,
            last_used_at=row.last_used_at,
        )

    def live_tokens(session: Session, user_id: str, now: datetime) -> list[ApiToken]:
        """The account's unexpired tokens, newest first; its expired ones are deleted."""
        session.execute(
            delete(ApiToken).where(
                ApiToken.user_id == user_id, ApiToken.expires_at <= now
            )
        )
        return list(
            session.scalars(
                select(ApiToken)
                .where(ApiToken.user_id == user_id)
                .order_by(ApiToken.created_at.desc())
            )
        )

    @router.get("/auth/tokens", response_model=list[ApiTokenInfo], tags=["auth"])
    def list_tokens(
        user: User = Depends(session_user), session: Session = Depends(get_session)
    ) -> list[ApiTokenInfo]:
        """The account's personal API tokens that have not expired, newest first."""
        rows = live_tokens(session, user.id, clock())
        session.commit()
        return [token_info(row) for row in rows]

    @router.post(
        "/auth/tokens", response_model=ApiTokenCreated, status_code=201, tags=["auth"]
    )
    def create_token(
        body: ApiTokenRequest,
        user: User = Depends(session_user),
        session: Session = Depends(get_session),
    ) -> ApiTokenCreated:
        """Create a personal API token for submitting from a script.

        The token is in this response and nowhere else: only its sha256 is stored.
        Send it as ``Authorization: Bearer <token>``. It expires after
        ``expires_in_days``. Needs the session token of a signed-in browser; an API
        token cannot create another.
        """
        days = body.expires_in_days or config.api_token_default_days
        if days > config.api_token_max_days:
            raise HTTPException(
                status_code=422,
                detail=f"a token lasts at most {config.api_token_max_days} days",
            )
        now = clock()
        session.execute(select(User.id).where(User.id == user.id).with_for_update())
        if len(live_tokens(session, user.id, now)) >= config.max_api_tokens:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"an account holds at most {config.max_api_tokens} API tokens; "
                    "revoke one first"
                ),
            )
        token, token_hash, hint = new_api_token()
        row = ApiToken(
            id=new_id(),
            user_id=user.id,
            name=body.name,
            token_sha256=token_hash,
            token_hint=hint,
            created_at=now,
            expires_at=now + timedelta(days=days),
        )
        session.add(row)
        session.commit()
        return ApiTokenCreated(**token_info(row).model_dump(), token=token)

    @router.post(
        "/auth/tokens/{token_id}/revoke", response_model=Message, tags=["auth"]
    )
    def revoke_token(
        token_id: str,
        user: User = Depends(session_user),
        session: Session = Depends(get_session),
    ) -> Message:
        """Revoke one of the account's personal API tokens; it stops working at once.

        The row is deleted, not marked: nothing about a revoked token is kept.
        """
        row = session.get(ApiToken, token_id)
        if row is None or row.user_id != user.id:
            raise HTTPException(status_code=404, detail="unknown API token")
        name = row.name
        session.delete(row)
        session.commit()
        return Message(message=f"API token {name} revoked")

    # --------------------------------------------------------------------- datasets

    @router.get(
        "/datasets", response_model=list[BenchmarkDatasetPublic], tags=["datasets"]
    )
    def list_datasets() -> list[BenchmarkDatasetPublic]:
        """Every benchmark dataset: targets, split sizes, primary metric, file hashes."""
        return [bundle.dataset.public() for bundle in bundles.values()]

    @router.get(
        "/datasets/{slug}", response_model=BenchmarkDatasetPublic, tags=["datasets"]
    )
    def get_dataset(slug: str) -> BenchmarkDatasetPublic:
        """One benchmark dataset."""
        return get_bundle(slug).dataset.public()

    @router.get("/datasets/{slug}/splits.csv", tags=["datasets"])
    def get_splits(slug: str) -> FileResponse:
        """``record_id,split`` for every record; ``X-Artifact-SHA256`` carries its hash."""
        bundle = get_bundle(slug)
        return FileResponse(
            bundle.root / SPLITS_FILENAME,
            media_type="text/csv",
            filename=f"{slug}-splits.csv",
            headers={"X-Artifact-SHA256": bundle.dataset.splits_sha256},
        )

    @router.get("/datasets/{slug}/template.csv", tags=["datasets"])
    def get_template(slug: str) -> FileResponse:
        """The submission template: every row to fill, with an empty ``prediction``."""
        bundle = get_bundle(slug)
        return FileResponse(
            bundle.root / TEMPLATE_FILENAME,
            media_type="text/csv",
            filename=f"{slug}-template.csv",
            headers={"X-Artifact-SHA256": bundle.dataset.template_sha256},
        )

    # ------------------------------------------------------------------ submissions

    @router.get("/quota", response_model=Quota, tags=["submissions"])
    def quota(
        user: User = Depends(current_user), session: Session = Depends(get_session)
    ) -> Quota:
        """How many attempts the account has left, and when the next is allowed."""
        return to_quota(quota_state(session, user.id, clock()))

    @router.post(
        "/submissions",
        response_model=SubmissionResult,
        status_code=201,
        tags=["submissions"],
    )
    def submit(
        dataset: str = Form(description="Slug of the benchmark dataset."),
        metadata: str = Form(description="SubmissionMetadata as a JSON string."),
        predictions: UploadFile = File(description="The predictions CSV."),
        user: User = Depends(current_user),
        session: Session = Depends(get_session),
    ) -> SubmissionResult:
        """Upload predictions for grading.

        Returns 201 with validation and test scores, 422 with the rejection reasons
        under ``detail``, or 429 with ``next_allowed_at`` when the quota is spent. A
        rejected upload counts as an attempt; validate locally first with
        ``python -m torchcell.benchmark.validation``.
        """
        if config.require_approval and not user.approved:
            raise HTTPException(status_code=403, detail="this account awaits approval")
        bundle = get_bundle(dataset)
        now = clock()
        # Serialize this account's attempts so two uploads cannot both pass the quota.
        session.execute(select(User.id).where(User.id == user.id).with_for_update())
        state = quota_state(session, user.id, now)
        if state.next_allowed_at is not None:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": state.reason,
                    "next_allowed_at": state.next_allowed_at.isoformat(),
                },
            )
        raw = predictions.file.read(config.max_upload_bytes + 1)
        submission = process_submission(
            session, user, bundle, metadata, raw, now, is_baseline=False
        )
        session.commit()
        return respond(submission)

    @router.get(
        "/submissions/mine", response_model=list[SubmissionResult], tags=["submissions"]
    )
    def my_submissions(
        user: User = Depends(current_user), session: Session = Depends(get_session)
    ) -> list[SubmissionResult]:
        """Every attempt of the account, newest first, rejected ones included."""
        rows = session.scalars(
            select(Submission)
            .where(Submission.user_id == user.id)
            .order_by(Submission.created_at.desc())
        ).all()
        return [_result(row) for row in rows]

    # ------------------------------------------------------------------ leaderboard

    @router.get(
        "/leaderboard/{slug}", response_model=list[LeaderboardRow], tags=["leaderboard"]
    )
    def leaderboard(
        slug: str, verified_only: bool = False, session: Session = Depends(get_session)
    ) -> list[LeaderboardRow]:
        """Scored submissions on one dataset, best test score on the primary metric first.

        ``verified_only=true`` keeps reproduced submissions only; the default also
        lists provisional ones, each marked by its ``status``.
        """
        bundle = get_bundle(slug)
        statuses = (SubmissionStatus.VERIFIED,) if verified_only else BOARD_STATUSES
        rows = session.execute(
            select(Submission, User)
            .join(User, Submission.user_id == User.id)
            .where(Submission.dataset_slug == slug, Submission.status.in_(statuses))
        ).all()
        primary = bundle.dataset.primary_metric
        board = [_row(submission, user) for submission, user in rows]
        board.sort(
            key=lambda row: oriented(primary, getattr(row.test.macro, primary)),
            reverse=True,
        )
        return [LeaderboardRow.model_validate(row.model_dump()) for row in board]

    @router.get(
        "/users/{user_id}/submissions", response_model=UserHistory, tags=["leaderboard"]
    )
    def user_history(
        user_id: str, session: Session = Depends(get_session)
    ) -> UserHistory:
        """An account's scored submissions on every dataset, oldest first."""
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="unknown user")
        rows = session.scalars(
            select(Submission)
            .where(Submission.user_id == user_id, Submission.status.in_(BOARD_STATUSES))
            .order_by(Submission.created_at)
        ).all()
        return UserHistory(
            user=_user_public(user), submissions=[_row(row, user) for row in rows]
        )

    # ------------------------------------------------------------------------ admin

    def review(
        session: Session,
        submission_id: str,
        allowed_from: tuple[SubmissionStatus, ...],
        new_status: SubmissionStatus,
        reviewer: str,
        note: str,
    ) -> SubmissionResult:
        submission = session.get(Submission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="unknown submission")
        if submission.status not in allowed_from:
            raise HTTPException(
                status_code=409, detail=f"submission is {submission.status}"
            )
        submission.status = new_status
        submission.reviewed_at = clock()
        submission.reviewed_by = reviewer
        submission.review_note = note
        session.commit()
        return _result(submission)

    @router.post(
        "/admin/submissions/{submission_id}/verify",
        response_model=SubmissionResult,
        tags=["admin"],
    )
    def verify_submission(
        submission_id: str,
        body: ReviewRequest,
        reviewer: str = Depends(require_admin),
        session: Session = Depends(get_session),
    ) -> SubmissionResult:
        """Mark a provisional submission as reproduced."""
        return review(
            session,
            submission_id,
            (SubmissionStatus.PROVISIONAL,),
            SubmissionStatus.VERIFIED,
            reviewer,
            body.note,
        )

    @router.post(
        "/admin/submissions/{submission_id}/withdraw",
        response_model=SubmissionResult,
        tags=["admin"],
    )
    def withdraw_submission(
        submission_id: str,
        body: ReviewRequest,
        reviewer: str = Depends(require_admin),
        session: Session = Depends(get_session),
    ) -> SubmissionResult:
        """Take a scored submission off the board; its row and archive are kept."""
        return review(
            session,
            submission_id,
            BOARD_STATUSES,
            SubmissionStatus.WITHDRAWN,
            reviewer,
            body.note,
        )

    def set_user_flags(session: Session, user_id: str, **flags: bool) -> Message:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="unknown user")
        for name, value in flags.items():
            setattr(user, name, value)
        session.commit()
        return Message(message=f"user {user_id} updated")

    @router.post(
        "/admin/users/{user_id}/approve", response_model=Message, tags=["admin"]
    )
    def approve_user(
        user_id: str,
        _: str = Depends(require_admin),
        session: Session = Depends(get_session),
    ) -> Message:
        """Allow an account to submit (used when new accounts are held for approval)."""
        return set_user_flags(session, user_id, approved=True)

    @router.post(
        "/admin/users/{user_id}/disable", response_model=Message, tags=["admin"]
    )
    def disable_user(
        user_id: str,
        _: str = Depends(require_admin),
        session: Session = Depends(get_session),
    ) -> Message:
        """Disable an account: its tokens stop working and it cannot sign in again."""
        return set_user_flags(session, user_id, disabled=True)

    @router.post(
        "/admin/baselines",
        response_model=SubmissionResult,
        status_code=201,
        tags=["admin"],
    )
    def submit_baseline(
        dataset: str = Form(),
        metadata: str = Form(),
        predictions: UploadFile = File(),
        _: str = Depends(require_admin),
        session: Session = Depends(get_session),
    ) -> SubmissionResult:
        """Grade a baseline's predictions through the same grader, outside the quota."""
        bundle = get_bundle(dataset)
        now = clock()
        system_user = session.scalar(
            select(User).where(User.email_canonical == BASELINE_USER_EMAIL)
        )
        if system_user is None:
            system_user = User(
                id=new_id(),
                email=BASELINE_USER_EMAIL,
                email_canonical=BASELINE_USER_EMAIL,
                display_name=BASELINE_USER_NAME,
                approved=True,
                is_system=True,
                created_at=now,
            )
            session.add(system_user)
            session.flush()
        raw = predictions.file.read(config.max_upload_bytes + 1)
        submission = process_submission(
            session, system_user, bundle, metadata, raw, now, is_baseline=True
        )
        session.commit()
        return respond(submission)

    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory torchcell.benchmark.app:create_app_from_env``."""
    load_dotenv()
    return create_app(BenchServerConfig.from_env())


def main() -> None:
    """CLI: run the server, ``--init-db`` to create tables, ``--gen-admin-key NAME``."""
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gen-admin-key", metavar="NAME", help="Mint an admin key, exit."
    )
    parser.add_argument(
        "--init-db", action="store_true", help="Create the tables, exit."
    )
    parser.add_argument("--host", default=None, help="Override TC_BENCH_HOST.")
    parser.add_argument(
        "--port", type=int, default=None, help="Override TC_BENCH_PORT."
    )
    args = parser.parse_args()

    if args.gen_admin_key:
        print_minted_key(args.gen_admin_key, ADMIN_KEYS_FILE_VAR)
        return

    config = BenchServerConfig.from_env()
    if args.init_db:
        init_schema(make_engine(config.database_url.get_secret_value()))
        print(json.dumps({"initialized": True}))
        return

    host = args.host or config.host
    port = config.port if args.port is None else args.port
    log.info(
        "benchmark endpoint: datasets %s on %s:%d", config.datasets_root, host, port
    )
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
