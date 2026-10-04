# torchcell/benchmark/client.py
# [[torchcell.benchmark.client]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/client.py
# Test file: tests/torchcell/benchmark/test_client.py

"""Client for the ``tc-bench`` endpoint: submit predictions from a script.

``BenchClient`` talks to :mod:`torchcell.benchmark.app` over httpx. Reading needs no
account: the dataset list and each dataset's template are public, so a client built
without a token can fetch them and validate a predictions file. The quota, a
submission and the list of attempts need a personal API token
(``Authorization: Bearer tcb_...``), which the account's owner creates on the
website's account page; calling one of them without a token raises
:class:`TokenRequired` before any request is sent. ``submit`` calls the same ``POST /submissions`` the website's form
calls, so a submission made here and one made in the browser are the same request.

``submit`` validates the predictions file against the dataset's template before it
uploads (the same :func:`torchcell.benchmark.validation.validate_predictions` the
server runs), because a rejected upload counts toward the quota and a file that fails
here never leaves the machine. It returns the :class:`SubmissionResult` both when the
attempt was scored and when the server rejected it, and raises :class:`QuotaExceeded`
when the quota is spent (HTTP 429, which is not an attempt).

Environment: ``TC_BENCH_URL`` (the API base, ending in ``/api/v1``) and, for the
calls that need an account, ``TC_BENCH_TOKEN`` (``BenchClient.from_env``). The ``tc-bench`` command wraps the
client: ``datasets``, ``template``, ``quota``, ``submit`` and ``mine``, each printing
JSON. ``submit`` exits 0 when scored, 1 when rejected and 2 when rate limited.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Self

import httpx
from dotenv import load_dotenv
from pydantic import TypeAdapter

from torchcell.benchmark.bundle import (
    BenchmarkDatasetPublic,
    SubmissionSpec,
    sha256_bytes,
)
from torchcell.benchmark.results import Quota, SubmissionResult
from torchcell.benchmark.submission import SubmissionMetadata
from torchcell.benchmark.validation import ValidationReport, validate_predictions

URL_VAR = "TC_BENCH_URL"
TOKEN_VAR = "TC_BENCH_TOKEN"
SHA256_HEADER = "X-Artifact-SHA256"
DEFAULT_TIMEOUT = 120.0

_DATASETS = TypeAdapter(list[BenchmarkDatasetPublic])
_RESULTS = TypeAdapter(list[SubmissionResult])


class HttpClient(Protocol):
    """The two ``httpx.Client`` calls the client makes; a test client satisfies it too."""

    def get(self, url: str, *, headers: Mapping[str, str]) -> Any:
        """A buffered GET returning a response with ``raise_for_status`` and ``content``."""
        ...

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        data: Mapping[str, str],
        files: Mapping[str, tuple[str, bytes, str]],
    ) -> Any:
        """A multipart POST returning a response with ``status_code`` and ``json``."""
        ...


class TokenRequired(RuntimeError):
    """A call that needs an account was made by a client that holds no API token."""


class TemplateIntegrityError(RuntimeError):
    """A downloaded template's sha256 does not match the header the server sent."""


class InvalidPredictions(ValueError):
    """The predictions file failed local validation and was not uploaded."""

    def __init__(self, report: ValidationReport) -> None:
        """Keep the validation ``report``; its reasons are the message."""
        super().__init__("; ".join(report.reasons))
        self.report = report


class QuotaExceeded(RuntimeError):
    """The server refused the upload because the account's quota is spent (HTTP 429)."""

    def __init__(self, message: str, next_allowed_at: datetime) -> None:
        """Keep the server's ``message`` and the time the next attempt is allowed."""
        super().__init__(message)
        self.next_allowed_at = next_allowed_at


class BenchClient:
    """HTTP client for one ``tc-bench`` endpoint; a token is needed only to submit."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        http: HttpClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """``url`` is the API base ending in ``/api/v1``; ``http`` injects a client."""
        self.url = url.rstrip("/")
        self._token = token
        self._http: HttpClient = (
            http if http is not None else httpx.Client(timeout=timeout)
        )

    @classmethod
    def from_env(cls, http: HttpClient | None = None) -> Self:
        """Build from ``TC_BENCH_URL`` (required) and ``TC_BENCH_TOKEN`` (optional)."""
        return cls(os.environ[URL_VAR], os.environ.get(TOKEN_VAR), http=http)

    def _auth(self) -> dict[str, str]:
        if self._token is None:
            raise TokenRequired(
                f"this call needs a personal API token; set {TOKEN_VAR} "
                "(create one on the website's account page)"
            )
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path: str, *, auth: bool = False) -> Any:
        response = self._http.get(
            f"{self.url}{path}", headers=self._auth() if auth else {}
        )
        response.raise_for_status()
        return response

    def datasets(self) -> list[BenchmarkDatasetPublic]:
        """Every benchmark dataset the endpoint serves. Needs no token."""
        return _DATASETS.validate_json(self._get("/datasets").content)

    def template(self, slug: str) -> bytes:
        """The template CSV of ``slug``, verified against its sha256. Needs no token."""
        response = self._get(f"/datasets/{slug}/template.csv")
        expected = response.headers[SHA256_HEADER]
        if sha256_bytes(response.content) != expected:
            raise TemplateIntegrityError(
                f"template of {slug} does not match its sha256 {expected}"
            )
        return bytes(response.content)

    def quota(self) -> Quota:
        """How many attempts the account has left and when the next is allowed."""
        return Quota.model_validate_json(self._get("/quota", auth=True).content)

    def mine(self) -> list[SubmissionResult]:
        """Every attempt of the account, newest first, rejected ones included."""
        return _RESULTS.validate_json(self._get("/submissions/mine", auth=True).content)

    def validate(self, slug: str, predictions: bytes) -> ValidationReport:
        """Validate ``predictions`` against the template of ``slug``; nothing is uploaded."""
        spec = SubmissionSpec.from_template_csv(self.template(slug).decode("utf-8"))
        report, _ = validate_predictions(predictions, spec)
        return report

    def submit(
        self,
        slug: str,
        metadata: SubmissionMetadata,
        predictions: Path,
        *,
        validate: bool = True,
    ) -> SubmissionResult:
        """Upload ``predictions`` for grading on dataset ``slug``.

        With ``validate`` (the default) a file that fails local validation raises
        :class:`InvalidPredictions` and is not uploaded. A rejection by the server is
        returned, not raised: read ``status`` and ``rejection_reasons``.
        """
        headers = self._auth()
        raw = predictions.read_bytes()
        if validate:
            report = self.validate(slug, raw)
            if not report.ok:
                raise InvalidPredictions(report)
        response = self._http.post(
            f"{self.url}/submissions",
            headers=headers,
            data={"dataset": slug, "metadata": metadata.model_dump_json()},
            files={"predictions": (predictions.name, raw, "text/csv")},
        )
        if response.status_code == 429:
            detail = response.json()["detail"]
            raise QuotaExceeded(
                detail["message"], datetime.fromisoformat(detail["next_allowed_at"])
            )
        if response.status_code == 422:
            return SubmissionResult.model_validate(response.json()["detail"])
        response.raise_for_status()
        return SubmissionResult.model_validate_json(response.content)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: list datasets, fetch a template, read the quota, submit, list attempts."""
    load_dotenv()
    parser = argparse.ArgumentParser(prog="tc-bench", description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("datasets", help="List the benchmark datasets.")
    template = commands.add_parser("template", help="Download a dataset's template.")
    template.add_argument("dataset")
    template.add_argument("--out", type=Path, required=True)
    commands.add_parser("quota", help="Show the account's quota.")
    commands.add_parser("mine", help="List the account's attempts.")
    submit = commands.add_parser("submit", help="Validate, upload and grade.")
    submit.add_argument("--dataset", required=True)
    submit.add_argument("--metadata", type=Path, required=True, help="A JSON file.")
    submit.add_argument("--predictions", type=Path, required=True, help="The CSV.")
    submit.add_argument(
        "--no-validate",
        action="store_true",
        help="Upload without validating locally first.",
    )
    args = parser.parse_args(argv)
    client = BenchClient.from_env()

    if args.command == "datasets":
        _print([d.model_dump(mode="json") for d in client.datasets()])
    elif args.command == "template":
        args.out.write_bytes(client.template(args.dataset))
        _print({"written": str(args.out)})
    elif args.command == "quota":
        _print(client.quota().model_dump(mode="json"))
    elif args.command == "mine":
        _print([r.model_dump(mode="json") for r in client.mine()])
    else:
        metadata = SubmissionMetadata.model_validate_json(
            args.metadata.read_text(encoding="utf-8")
        )
        try:
            result = client.submit(
                args.dataset, metadata, args.predictions, validate=not args.no_validate
            )
        except InvalidPredictions as invalid:
            _print({"uploaded": False, **invalid.report.model_dump(mode="json")})
            sys.exit(1)
        except QuotaExceeded as refused:
            _print(
                {
                    "uploaded": False,
                    "message": str(refused),
                    "next_allowed_at": refused.next_allowed_at.isoformat(),
                }
            )
            sys.exit(2)
        _print(result.model_dump(mode="json"))
        sys.exit(1 if result.rejection_reasons else 0)


if __name__ == "__main__":
    main()
