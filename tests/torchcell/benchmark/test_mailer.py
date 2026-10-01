# tests/torchcell/benchmark/test_mailer.py
# [[tests.torchcell.benchmark.test_mailer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_mailer.py
"""``torchcell.benchmark.mailer`` without a network.

``smtplib.SMTP`` is replaced by a recorder, so the test asserts the order of the SMTP
conversation (connect with the configured host, port and 20 s timeout; STARTTLS with a
context; login; one message) and the headers of the message that would be sent.
"""

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

import pytest
from pydantic import SecretStr

from torchcell.benchmark.mailer import (
    ConsoleMailer,
    MemoryMailer,
    SmtpConfig,
    SmtpMailer,
    confirmation_message,
)


class RecordingSmtp:
    """Stands in for ``smtplib.SMTP`` and records every call."""

    calls: list[tuple[str, Any]] = []

    def __init__(self, host: str, port: int, timeout: float) -> None:
        """Record the connection arguments."""
        self.calls.append(("connect", (host, port, timeout)))

    def __enter__(self) -> "RecordingSmtp":
        """Enter the connection context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Record the connection closing."""
        self.calls.append(("quit", None))

    def starttls(self, context: ssl.SSLContext) -> None:
        """Record the TLS upgrade and its context."""
        self.calls.append(("starttls", context))

    def login(self, username: str, password: str) -> None:
        """Record the credentials."""
        self.calls.append(("login", (username, password)))

    def send_message(self, message: EmailMessage) -> None:
        """Record the message."""
        self.calls.append(("send", message))


def test_smtp_mailer_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingSmtp.calls = []
    monkeypatch.setattr(smtplib, "SMTP", RecordingSmtp)
    config = SmtpConfig(
        host="smtp.example.org",
        username="bench",
        password=SecretStr("smtp-password"),
        sender="noreply@example.org",
    )
    mailer = SmtpMailer(config)
    assert mailer.config is config
    mailer.send("alice@example.org", "Subject line", "Body text\n")

    names = [name for name, _ in RecordingSmtp.calls]
    assert names == ["connect", "starttls", "login", "send", "quit"]
    assert RecordingSmtp.calls[0][1] == ("smtp.example.org", 587, 20)
    context = RecordingSmtp.calls[1][1]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert RecordingSmtp.calls[2][1] == ("bench", "smtp-password")
    message = RecordingSmtp.calls[3][1]
    assert message["From"] == "noreply@example.org"
    assert message["To"] == "alice@example.org"
    assert message["Subject"] == "Subject line"
    assert message.get_content() == "Body text\n"


def test_smtp_password_is_not_in_the_repr() -> None:
    config = SmtpConfig(
        host="h",
        username="u",
        password=SecretStr("smtp-password"),
        sender="s@example.org",
    )
    assert "smtp-password" not in repr(config)


def test_memory_mailer_keeps_messages() -> None:
    mailer = MemoryMailer()
    mailer.send("a@example.org", "s1", "b1")
    mailer.send("b@example.org", "s2", "b2")
    assert mailer.outbox == [
        ("a@example.org", "s1", "b1"),
        ("b@example.org", "s2", "b2"),
    ]


def test_console_mailer_logs_the_message(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="torchcell.benchmark.mailer"):
        ConsoleMailer().send("a@example.org", "Confirm", "the link")
    assert caplog.messages == ["mail to a@example.org | Confirm\nthe link"]


def test_confirmation_message() -> None:
    subject, body = confirmation_message(
        "Alice", "https://site.example/benchmark/account?verify=abc", 24
    )
    assert subject == "Confirm your TorchCell benchmark account"
    assert body.startswith("Hello Alice,\n\n")
    assert "\n\nhttps://site.example/benchmark/account?verify=abc\n\n" in body
    assert "valid for 24 hours and works once" in body
