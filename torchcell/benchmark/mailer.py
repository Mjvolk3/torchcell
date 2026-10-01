# torchcell/benchmark/mailer.py
# [[torchcell.benchmark.mailer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/mailer.py
# Test file: tests/torchcell/benchmark/test_mailer.py

"""Outgoing mail for account confirmation.

Three senders share one method, ``send(to, subject, body)``:

- :class:`SmtpMailer`: the real one, over STARTTLS with a verified certificate.
- :class:`ConsoleMailer`: writes the message to the log. It exists for local
  development, where the confirmation link is read from the terminal, and is only used
  when the server is explicitly configured with ``TC_BENCH_EMAIL_BACKEND=console``.
- :class:`MemoryMailer`: keeps messages in a list, for tests.

A send that fails raises; the signup request then fails and the account is not created,
so no account exists whose confirmation mail was never sent.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr

log = logging.getLogger(__name__)

SMTP_TIMEOUT_SECONDS = 20


class Mailer(Protocol):
    """Anything that can send one plain-text message."""

    def send(self, to: str, subject: str, body: str) -> None:
        """Send ``body`` to ``to``."""
        ...


class SmtpConfig(BaseModel):
    """Connection settings for the SMTP submission port."""

    model_config = ConfigDict(frozen=True)

    host: str
    port: int = 587
    username: str
    password: SecretStr
    sender: str = Field(
        description="The From address, for example noreply@example.org."
    )


class SmtpMailer:
    """Sends through an SMTP server over STARTTLS."""

    def __init__(self, config: SmtpConfig) -> None:
        """Bind the connection settings; no connection is opened until ``send``."""
        self.config = config

    def send(self, to: str, subject: str, body: str) -> None:
        """Open a connection, upgrade it to TLS, authenticate, and send."""
        message = EmailMessage()
        message["From"] = self.config.sender
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP(
            self.config.host, self.config.port, timeout=SMTP_TIMEOUT_SECONDS
        ) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(self.config.username, self.config.password.get_secret_value())
            smtp.send_message(message)


class ConsoleMailer:
    """Logs the message instead of sending it (local development only)."""

    def send(self, to: str, subject: str, body: str) -> None:
        """Write the message to the log at INFO."""
        log.info("mail to %s | %s\n%s", to, subject, body)


class MemoryMailer:
    """Keeps every message in ``outbox`` (tests)."""

    def __init__(self) -> None:
        """Start with an empty outbox."""
        self.outbox: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, body: str) -> None:
        """Append ``(to, subject, body)`` to the outbox."""
        self.outbox.append((to, subject, body))


def confirmation_message(
    display_name: str, link: str, hours_valid: int
) -> tuple[str, str]:
    """Subject and body of the address-confirmation mail."""
    subject = "Confirm your TorchCell benchmark account"
    body = (
        f"Hello {display_name},\n\n"
        "Open this link to confirm your address and activate your TorchCell benchmark "
        f"account. It is valid for {hours_valid} hours and works once.\n\n"
        f"{link}\n\n"
        "If you did not sign up, ignore this message and no account will be activated.\n"
    )
    return subject, body
