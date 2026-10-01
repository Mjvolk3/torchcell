# tests/torchcell/benchmark/test_app.py
# [[tests.torchcell.benchmark.test_app]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_app.py
"""``torchcell.benchmark.app`` (tc-bench) end to end, in process.

The app runs on a SQLite file, the ``toy-fitness`` bundle of ``conftest.py``, a
``MemoryMailer`` whose outbox holds the confirmation links, and a settable clock (the
quota reads it; session tokens use the wall clock). Scores are asserted against the
hand-worked values: the labels themselves score Pearson 1 on both splits, and the
validation predictions 2, 1, 3, 4 against the labels 1, 2, 3, 4 score Pearson 0.8.

Covered: the signup, confirm and sign-in sequence and each way it is refused; one
account per canonical address; lockout after five wrong passwords; the public dataset
routes and their sha256 header; a scored submission, its archive and its board row; a
rejected submission and its reasons; the quota (one hour apart, three per 24 hours,
rejected attempts counted); the upload size guard; the integrity flag; admin verify,
withdraw, approve, disable and baselines; and ``BenchServerConfig.from_env``.
"""

import io
import json
import re
import sys
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from sqlalchemy import create_engine, select

from torchcell.api_keys import ApiKeys
from torchcell.benchmark import app as app_module
from torchcell.benchmark.app import (
    API_PREFIX,
    SIGNUP_MESSAGE,
    BenchServerConfig,
    build_mailer,
    create_app,
)
from torchcell.benchmark.db import User, init_schema, make_session_factory
from torchcell.benchmark.mailer import ConsoleMailer, MemoryMailer, SmtpMailer

SLUG = "toy-fitness"
ADMIN = {"X-API-Key": "admin-key-123"}
PASSWORD = "a-long-passphrase"
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
        self.config = BenchServerConfig(
            database_url=SecretStr("postgresql+psycopg://unused"),
            datasets_root=datasets_root,
            submissions_root=self.submissions_root,
            jwt_secret=SecretStr("j" * 40),
            admin_keys=ApiKeys.from_pairs("ops:admin-key-123"),
            account_url="https://site.example/benchmark/account",
            cors_origins=("https://site.example",),
            email_backend="console",
            **overrides,
        )
        self.engine = create_engine(
            f"sqlite:///{tmp_path / 'bench.db'}",
            connect_args={"check_same_thread": False},
        )
        init_schema(self.engine)
        self.sessions = make_session_factory(self.engine)
        self.mailer = MemoryMailer()
        self.clock = Clock()
        self.client = TestClient(
            create_app(
                self.config, engine=self.engine, mailer=self.mailer, clock=self.clock
            )
        )

    def url(self, path: str) -> str:
        """The full URL path of an API route."""
        return f"{API_PREFIX}{path}"

    def last_token(self) -> str:
        """The confirmation token in the newest mail."""
        match = re.search(r"\?verify=(\S+)", self.mailer.outbox[-1][2])
        assert match is not None
        return match.group(1)

    def register(
        self, email: str = "alice@example.org", name: str = "Alice"
    ) -> dict[str, str]:
        """Sign up, confirm, sign in; return the bearer header."""
        signup = self.client.post(
            self.url("/auth/signup"),
            json={"email": email, "password": PASSWORD, "display_name": name},
        )
        assert signup.status_code == 201
        verify = self.client.post(
            self.url("/auth/verify"), json={"token": self.last_token()}
        )
        assert verify.status_code == 200
        login = self.client.post(
            self.url("/auth/login"), json={"email": email, "password": PASSWORD}
        )
        assert login.status_code == 200
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

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


def test_signup_confirm_login_sequence(bench: Bench) -> None:
    body = {"email": "Alice@Example.org", "password": PASSWORD, "display_name": "Alice"}
    signup = bench.client.post(bench.url("/auth/signup"), json=body)
    assert signup.status_code == 201
    assert signup.json() == {"message": SIGNUP_MESSAGE}
    to, subject, text = bench.mailer.outbox[0]
    assert to == "alice@example.org"
    assert subject == "Confirm your TorchCell benchmark account"
    assert "https://site.example/benchmark/account?verify=" in text

    credentials = {"email": "alice@example.org", "password": PASSWORD}
    before = bench.client.post(bench.url("/auth/login"), json=credentials)
    assert before.status_code == 403
    assert before.json() == {"detail": "confirm your email address first"}

    token = bench.last_token()
    assert bench.client.post(
        bench.url("/auth/verify"), json={"token": token}
    ).json() == {"message": "address confirmed; you can sign in"}
    reused = bench.client.post(bench.url("/auth/verify"), json={"token": token})
    assert reused.status_code == 400
    assert reused.json() == {"detail": "invalid or expired token"}

    login = bench.client.post(bench.url("/auth/login"), json=credentials)
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    me = bench.client.get(
        bench.url("/auth/me"),
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    ).json()
    assert me["email"] == "alice@example.org"
    assert me["display_name"] == "Alice"
    assert (me["email_verified"], me["approved"]) == (True, True)
    assert me["created_at"] == "2026-10-01T12:00:00Z"


def test_confirmation_token_expires(bench: Bench) -> None:
    bench.client.post(
        bench.url("/auth/signup"),
        json={"email": "a@example.org", "password": PASSWORD, "display_name": "Al"},
    )
    bench.clock.now = T0 + timedelta(hours=24)
    expired = bench.client.post(
        bench.url("/auth/verify"), json={"token": bench.last_token()}
    )
    assert expired.status_code == 400


@pytest.mark.parametrize(
    ("body", "field"),
    [
        (
            {"email": "not-an-email", "password": PASSWORD, "display_name": "Al"},
            "email",
        ),
        (
            {"email": "a@example.org", "password": "short", "display_name": "Al"},
            "password",
        ),
        (
            {"email": "a@example.org", "password": PASSWORD, "display_name": "A"},
            "display_name",
        ),
        (
            {
                "email": "a@example.org",
                "password": PASSWORD,
                "display_name": "Al",
                "is_admin": 1,
            },
            "is_admin",
        ),
    ],
)
def test_signup_body_validation(bench: Bench, body: dict[str, Any], field: str) -> None:
    response = bench.client.post(bench.url("/auth/signup"), json=body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]
    assert bench.mailer.outbox == []


def test_one_account_per_canonical_address(bench: Bench) -> None:
    bench.register("alice@gmail.com")
    again = bench.client.post(
        bench.url("/auth/signup"),
        json={
            "email": "a.lice+two@googlemail.com",
            "password": PASSWORD,
            "display_name": "Al",
        },
    )
    # Same answer as a fresh signup, no second account, no second mail.
    assert again.status_code == 201
    assert again.json() == {"message": SIGNUP_MESSAGE}
    assert len(bench.mailer.outbox) == 1
    with bench.sessions() as session:
        assert session.scalars(select(User.email_canonical)).all() == [
            "alice@gmail.com"
        ]


def test_unconfirmed_signup_can_be_repeated_three_times(bench: Bench) -> None:
    body = {"email": "a@example.org", "password": PASSWORD, "display_name": "Al"}
    for _ in range(5):
        assert (
            bench.client.post(bench.url("/auth/signup"), json=body).status_code == 201
        )
    # three live tokens at most, so the fourth and fifth signups send nothing
    assert len(bench.mailer.outbox) == 3


def test_signups_per_address_are_limited(bench: Bench) -> None:
    for i in range(5):
        body = {
            "email": f"user{i}@example.org",
            "password": PASSWORD,
            "display_name": "Al",
        }
        assert (
            bench.client.post(bench.url("/auth/signup"), json=body).status_code == 201
        )
    sixth = bench.client.post(
        bench.url("/auth/signup"),
        json={"email": "user5@example.org", "password": PASSWORD, "display_name": "Al"},
    )
    assert sixth.status_code == 429
    assert sixth.json() == {"detail": "too many signups; try tomorrow"}
    bench.clock.now = T0 + timedelta(hours=24, seconds=1)
    later = bench.client.post(
        bench.url("/auth/signup"),
        json={"email": "user5@example.org", "password": PASSWORD, "display_name": "Al"},
    )
    assert later.status_code == 201


def test_blocked_domain_cannot_register(tmp_path: Path, datasets_root: Path) -> None:
    from torchcell.benchmark.security import AccountPolicy

    bench = Bench(
        tmp_path,
        datasets_root,
        account_policy=AccountPolicy(blocked_domains=frozenset({"mailinator.com"})),
    )
    response = bench.client.post(
        bench.url("/auth/signup"),
        json={"email": "x@mailinator.com", "password": PASSWORD, "display_name": "Al"},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "addresses at mailinator.com cannot register"}


def test_login_failures_and_lockout(bench: Bench) -> None:
    bench.register()
    wrong = {"email": "alice@example.org", "password": "wrong-password-x"}
    unknown = bench.client.post(
        bench.url("/auth/login"),
        json={"email": "nobody@example.org", "password": PASSWORD},
    )
    assert unknown.status_code == 401
    assert unknown.json() == {"detail": "invalid email or password"}
    for _ in range(5):
        attempt = bench.client.post(bench.url("/auth/login"), json=wrong)
        assert attempt.status_code == 401
        assert attempt.json() == {"detail": "invalid email or password"}
    good = {"email": "alice@example.org", "password": PASSWORD}
    locked = bench.client.post(bench.url("/auth/login"), json=good)
    assert locked.status_code == 429
    assert locked.json() == {"detail": "too many failed sign-ins; try later"}
    bench.clock.now = T0 + timedelta(minutes=15)
    assert bench.client.post(bench.url("/auth/login"), json=good).status_code == 200


def test_routes_that_need_a_session(bench: Bench) -> None:
    for method, path in [
        ("GET", "/auth/me"),
        ("GET", "/quota"),
        ("GET", "/submissions/mine"),
    ]:
        response = bench.client.request(method, bench.url(path))
        assert response.status_code == 401
        assert response.json() == {"detail": "sign in to use this route"}
        assert response.headers["www-authenticate"] == "Bearer"
    forged = bench.client.get(
        bench.url("/auth/me"), headers={"Authorization": "Bearer x.y.z"}
    )
    assert forged.status_code == 401


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
    import hashlib

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
    login = bench.client.post(
        bench.url("/auth/login"),
        json={"email": "alice@example.org", "password": PASSWORD},
    )
    assert login.status_code == 403
    assert login.json() == {"detail": "this account is disabled"}
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
    # the system account cannot be signed in to
    login = bench.client.post(
        bench.url("/auth/login"),
        json={"email": "baselines@torchcell.invalid", "password": PASSWORD},
    )
    assert login.status_code in (401, 422)
    rejected = bench.submit(ADMIN, b"bad", knn, path="/admin/baselines")
    assert rejected.status_code == 422


# ------------------------------------------------------------------------- config


def _env(tmp_path: Path, datasets_root: Path) -> dict[str, str]:
    (tmp_path / "db_password").write_text("p@ss/word\n")
    (tmp_path / "jwt_secret").write_text("k" * 48 + "\n")
    (tmp_path / "smtp_password").write_text("smtp-secret\n")
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
        "TC_BENCH_ACCOUNT_URL": "https://site.example/benchmark/account",
        "TC_BENCH_CORS_ORIGINS": "https://site.example, https://mjvolk3.github.io",
        "TC_BENCH_EMAIL_BACKEND": "smtp",
        "TC_BENCH_SMTP_HOST": "smtp.example.org",
        "TC_BENCH_SMTP_USERNAME": "bench",
        "TC_BENCH_SMTP_PASSWORD_FILE": str(tmp_path / "smtp_password"),
        "TC_BENCH_SMTP_SENDER": "noreply@example.org",
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
    assert config.account_policy.blocked_domains == {"mailinator.com", "trashmail.com"}
    assert config.account_policy.allowed_domain_suffixes == (".edu", ".ac.uk")
    assert config.smtp is not None
    assert (config.smtp.host, config.smtp.port) == ("smtp.example.org", 587)
    assert config.smtp.password.get_secret_value() == "smtp-secret"
    assert config.admin_keys.hashes == {"ops": "0" * 64}
    assert "p@ss" not in repr(config) and "kkkk" not in repr(config)
    assert isinstance(build_mailer(config), SmtpMailer)


def test_config_console_backend_and_missing_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets_root: Path
) -> None:
    env = _env(tmp_path, datasets_root)
    console = {k: v for k, v in env.items() if "SMTP" not in k}
    console["TC_BENCH_EMAIL_BACKEND"] = "console"
    _set_env(monkeypatch, console)
    config = BenchServerConfig.from_env()
    assert config.smtp is None
    assert isinstance(build_mailer(config), ConsoleMailer)

    del console["TC_BENCH_JWT_SECRET_FILE"]
    _set_env(monkeypatch, console)
    with pytest.raises(KeyError, match="TC_BENCH_JWT_SECRET_FILE"):
        BenchServerConfig.from_env()


def test_config_refuses_a_short_secret_and_smtp_without_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets_root: Path
) -> None:
    env = _env(tmp_path, datasets_root)
    (tmp_path / "jwt_secret").write_text("short")
    _set_env(monkeypatch, env)
    with pytest.raises(ValidationError, match="jwt secret must be at least 32 bytes"):
        BenchServerConfig.from_env()
    with pytest.raises(
        ValidationError, match="email_backend 'smtp' needs the smtp settings"
    ):
        BenchServerConfig(
            database_url=SecretStr("postgresql+psycopg://unused"),
            datasets_root=datasets_root,
            submissions_root=tmp_path,
            jwt_secret=SecretStr("j" * 40),
            admin_keys=ApiKeys(hashes={}),
            account_url="https://site.example/account",
            cors_origins=(),
            email_backend="smtp",
        )


def test_trust_proxy_reads_the_forwarded_address(
    tmp_path: Path, datasets_root: Path
) -> None:
    bench = Bench(tmp_path, datasets_root, trust_proxy=True, max_signups_per_address=1)

    def signup(email: str, forwarded: str) -> int:
        response = bench.client.post(
            bench.url("/auth/signup"),
            json={"email": email, "password": PASSWORD, "display_name": "Al"},
            headers={"X-Forwarded-For": forwarded},
        )
        return int(response.status_code)

    assert signup("a@example.org", "198.51.100.9, 203.0.113.7") == 201
    # same proxy-appended address, different client-supplied prefix: limited
    assert signup("b@example.org", "10.0.0.1, 203.0.113.7") == 429
    assert signup("c@example.org", "203.0.113.8") == 201


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
    env = {k: v for k, v in _env(tmp_path, datasets_root).items() if "SMTP" not in k}
    env["TC_BENCH_EMAIL_BACKEND"] = "console"
    _set_env(monkeypatch, env)
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
        "email_tokens",
        "submissions",
        "users",
    ]

    monkeypatch.setattr(sys, "argv", ["tc-bench-server"])
    app_module.main()
    assert started == {"host": "127.0.0.1", "port": 8725}
    monkeypatch.setattr(sys, "argv", ["tc-bench-server", "--port", "9000"])
    app_module.main()
    assert started == {"host": "127.0.0.1", "port": 9000}
