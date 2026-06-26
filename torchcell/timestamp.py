"""Timestamp helper for naming generated files and image outputs."""

import datetime


def timestamp() -> str:
    """Return the current local time as a ``YYYY-MM-DD-HH-MM-SS`` string."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
