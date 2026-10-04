# tests/torchcell/benchmark/test_client.py
# [[tests.torchcell.benchmark.test_client]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_client.py
"""``torchcell.benchmark.client`` against the tc-bench app through the ASGI test client.

The client holds a personal API token created the way a person creates one: sign in,
then ``POST /auth/tokens``. Covered: reading without a token and the refusal of the
calls that need one; the dataset list and the template with its sha256
check; the quota; a scored submission (Pearson 1 from the labels themselves) and that
it is the same row the board shows; local validation stopping a malformed file before
it is uploaded, so no attempt is spent; a server-side rejection returned as a result;
the quota refusal raised with its time; a revoked token; and the ``tc-bench`` command's
output and exit codes.
"""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest

from tests.torchcell.benchmark.test_app import (
    METADATA,
    SERVICE,
    SLUG,
    T0,
    Bench,
    Csv,
    Labels,
)
from torchcell.benchmark import client as client_module
from torchcell.benchmark.app import API_PREFIX
from torchcell.benchmark.client import (
    BenchClient,
    InvalidPredictions,
    QuotaExceeded,
    TemplateIntegrityError,
    TokenRequired,
)
from torchcell.benchmark.db import SubmissionStatus
from torchcell.benchmark.submission import SubmissionMetadata

URL = f"{SERVICE}{API_PREFIX}"
META = SubmissionMetadata.model_validate(METADATA)


@pytest.fixture
def bench(tmp_path: Path, datasets_root: Path) -> Bench:
    return Bench(tmp_path, datasets_root)


@pytest.fixture
def api(bench: Bench) -> BenchClient:
    """A client holding a fresh personal API token of a new account."""
    created = bench.client.post(
        bench.url("/auth/tokens"), headers=bench.register(), json={"name": "ci"}
    )
    return BenchClient(f"{URL}/", created.json()["token"], http=bench.client)


def _write(tmp_path: Path, raw: bytes) -> Path:
    path = tmp_path / "predictions.csv"
    path.write_bytes(raw)
    return path


def test_datasets_template_and_quota(api: BenchClient, datasets_root: Path) -> None:
    assert api.url == URL  # the trailing slash is dropped
    (dataset,) = api.datasets()
    assert (dataset.slug, dataset.primary_metric) == (SLUG, "pearson")
    template = api.template(SLUG)
    assert template == (datasets_root / SLUG / "template.csv").read_bytes()
    quota = api.quota()
    assert (quota.max_per_window, quota.used_in_window, quota.remaining) == (3, 0, 3)
    assert quota.next_allowed_at is None
    assert api.mine() == []


def test_template_that_does_not_match_its_hash_is_refused(api: BenchClient) -> None:
    def tampered(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"record_id\n", headers={"X-Artifact-SHA256": "0" * 64}
        )

    broken = BenchClient(
        URL, http=httpx.Client(transport=httpx.MockTransport(tampered))
    )
    with pytest.raises(TemplateIntegrityError, match="toy-fitness"):
        broken.template(SLUG)


def test_submit_scores_and_reaches_the_board(
    bench: Bench, api: BenchClient, tmp_path: Path, labels: Labels, to_csv: Csv
) -> None:
    result = api.submit(SLUG, META, _write(tmp_path, to_csv(labels)))
    assert result.status is SubmissionStatus.PROVISIONAL
    assert result.rejection_reasons == []
    assert result.val is not None and result.test is not None
    assert result.val.macro.pearson == pytest.approx(1.0)
    assert result.test.macro.mse == 0.0
    board = bench.client.get(bench.url(f"/leaderboard/{SLUG}")).json()
    assert [row["submission_id"] for row in board] == [result.submission_id]
    assert [r.submission_id for r in api.mine()] == [result.submission_id]
    assert api.quota().used_in_window == 1


def test_local_validation_stops_a_bad_file_before_upload(
    api: BenchClient, tmp_path: Path, labels: Labels, to_csv: Csv
) -> None:
    del labels[("s4", "fitness")]
    path = _write(tmp_path, to_csv(labels))
    report = api.validate(SLUG, path.read_bytes())
    assert not report.ok
    with pytest.raises(InvalidPredictions) as invalid:
        api.submit(SLUG, META, path)
    assert invalid.value.report.reasons == report.reasons
    assert any("s4" in reason for reason in report.reasons)
    assert api.quota().used_in_window == 0  # nothing was uploaded
    assert api.mine() == []


def test_server_rejection_is_returned_and_counts(
    api: BenchClient, tmp_path: Path, labels: Labels, to_csv: Csv
) -> None:
    del labels[("s4", "fitness")]
    result = api.submit(SLUG, META, _write(tmp_path, to_csv(labels)), validate=False)
    assert result.status is SubmissionStatus.REJECTED
    assert any("s4" in reason for reason in result.rejection_reasons)
    assert (result.val, result.test) == (None, None)
    assert api.quota().used_in_window == 1


def test_quota_refusal_is_raised_with_its_time(
    bench: Bench, api: BenchClient, tmp_path: Path, labels: Labels, to_csv: Csv
) -> None:
    path = _write(tmp_path, to_csv(labels))
    api.submit(SLUG, META, path)
    bench.clock.now = T0 + timedelta(minutes=30)
    with pytest.raises(QuotaExceeded) as refused:
        api.submit(SLUG, META, path)
    assert refused.value.next_allowed_at == T0 + timedelta(hours=1)
    assert api.quota().used_in_window == 1  # a 429 is not an attempt


def test_revoked_token_raises(bench: Bench, tmp_path: Path) -> None:
    headers = bench.register()
    created = bench.client.post(
        bench.url("/auth/tokens"), headers=headers, json={"name": "ci"}
    ).json()
    api = BenchClient(URL, created["token"], http=bench.client)
    assert api.quota().remaining == 3
    bench.client.post(
        bench.url(f"/auth/tokens/{created['token_id']}/revoke"), headers=headers
    )
    # The ASGI test client raises its own library's status error.
    with pytest.raises(httpx2.HTTPStatusError, match="401 Unauthorized"):
        api.quota()


def test_from_env(monkeypatch: pytest.MonkeyPatch, bench: Bench) -> None:
    monkeypatch.setenv("TC_BENCH_URL", URL)
    monkeypatch.setenv("TC_BENCH_TOKEN", "tcb_from_env")
    built = BenchClient.from_env(http=bench.client)
    assert built.url == URL
    assert built._auth() == {"Authorization": "Bearer tcb_from_env"}
    monkeypatch.delenv("TC_BENCH_TOKEN")
    assert BenchClient.from_env(http=bench.client)._token is None
    monkeypatch.delenv("TC_BENCH_URL")
    with pytest.raises(KeyError, match="TC_BENCH_URL"):
        BenchClient.from_env(http=bench.client)


def test_reading_needs_no_token_and_submitting_does(
    bench: Bench, tmp_path: Path, datasets_root: Path, labels: Labels, to_csv: Csv
) -> None:
    anonymous = BenchClient(URL, http=bench.client)
    assert [d.slug for d in anonymous.datasets()] == [SLUG]
    assert (
        anonymous.template(SLUG) == (datasets_root / SLUG / "template.csv").read_bytes()
    )
    assert anonymous.validate(SLUG, to_csv(labels)).ok
    path = _write(tmp_path, to_csv(labels))
    for call in (
        anonymous.quota,
        anonymous.mine,
        lambda: anonymous.submit(SLUG, META, path),
    ):
        with pytest.raises(TokenRequired, match="TC_BENCH_TOKEN"):
            call()


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    api: BenchClient,
    argv: list[str],
) -> tuple[int, Any]:
    """Run the ``tc-bench`` command on ``api``; return its exit code and printed JSON."""
    monkeypatch.setattr(BenchClient, "from_env", classmethod(lambda cls: api))
    monkeypatch.setattr(client_module, "load_dotenv", lambda: None)
    code = 0
    try:
        client_module.main(argv)
    except SystemExit as exit_:
        code = int(exit_.code or 0)
    return code, json.loads(capsys.readouterr().out)


def test_cli_submit_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    api: BenchClient,
    tmp_path: Path,
    labels: Labels,
    to_csv: Csv,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(METADATA))
    good = _write(tmp_path, to_csv(labels))
    argv = ["submit", "--dataset", SLUG, "--metadata", str(metadata)]

    bad = tmp_path / "bad.csv"
    bad.write_bytes(to_csv({k: v for k, v in labels.items() if k[0] != "s4"}))
    code, out = _run(monkeypatch, capsys, api, [*argv, "--predictions", str(bad)])
    assert code == 1
    assert out["uploaded"] is False and out["ok"] is False
    assert any("s4" in reason for reason in out["reasons"])

    code, out = _run(monkeypatch, capsys, api, [*argv, "--predictions", str(good)])
    assert (code, out["status"]) == (0, "provisional")
    assert out["val"]["macro"]["pearson"] == pytest.approx(1.0)

    code, out = _run(monkeypatch, capsys, api, [*argv, "--predictions", str(good)])
    assert code == 2
    assert out["uploaded"] is False
    assert out["next_allowed_at"] == "2026-10-01T13:00:00+00:00"

    code, out = _run(
        monkeypatch, capsys, api, [*argv, "--predictions", str(bad), "--no-validate"]
    )
    assert code == 2  # still inside the one-hour gap, so the bad file is not an attempt


def test_cli_read_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    api: BenchClient,
    tmp_path: Path,
    datasets_root: Path,
) -> None:
    code, out = _run(monkeypatch, capsys, api, ["datasets"])
    assert (code, [d["slug"] for d in out]) == (0, [SLUG])
    target = tmp_path / "template.csv"
    code, out = _run(monkeypatch, capsys, api, ["template", SLUG, "--out", str(target)])
    assert (code, out) == (0, {"written": str(target)})
    assert target.read_bytes() == (datasets_root / SLUG / "template.csv").read_bytes()
    code, out = _run(monkeypatch, capsys, api, ["quota"])
    assert (code, out["remaining"]) == (0, 3)
    code, out = _run(monkeypatch, capsys, api, ["mine"])
    assert (code, out) == (0, [])
