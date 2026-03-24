"""Kube-specific logging helpers.

This module exists to give both the CLI and library-adjacent tooling one
consistent logging surface. Kube often emits structured facts such as file
paths, shapes, shard counts, and sequence lengths, so the formatter is designed
to show short human-readable messages plus compact key-value context.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _coerce_level(level: str | int | None) -> int:
    """Normalize a string or numeric log level into a logging constant.

    This helper exists so CLI flags, environment variables, and direct Python
    calls can all feed the same configuration path without duplicating parsing
    logic.
    """
    if isinstance(level, int):
        return level
    if level is None:
        level = DEFAULT_LOG_LEVEL
    normalized = str(level).strip().lower()
    if normalized not in LOG_LEVELS:
        raise ValueError(f"unsupported log level {level!r}")
    return LOG_LEVELS[normalized]


def _stringify_field(value: Any) -> str:
    """Render structured field values into compact single-token log text.

    Kube logs frequently attach shapes, paths, booleans, and short sequences.
    This helper exists to keep those values readable and stable across command
    handlers.
    """
    if value is None:
        return "none"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, tuple) and len(value) == 3 and all(isinstance(item, int) for item in value):
        return "x".join(str(item) for item in value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify_field(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_stringify_field(item)}" for key, item in sorted(value.items())) + "}"
    return str(value).replace("\n", "\\n")


class KubeFormatter(logging.Formatter):
    """Format log records into a compact Kube-oriented terminal line.

    The formatter exists because generic logging output tends to hide the small
    structured facts that matter in Kube workflows, such as path, shape, shard,
    and sequence metadata.
    """

    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        """Render one record using Kube's timestamp, scope, and field layout."""
        timestamp = self.formatTime(record, "%H:%M:%S")
        scope = record.name
        if scope.startswith("kube."):
            scope = scope[5:]
        fields = getattr(record, "kube_fields", None) or {}
        field_text = ""
        if fields:
            field_text = " | " + " ".join(
                f"{key}={_stringify_field(value)}"
                for key, value in fields.items()
                if value is not None
            )
        return f"{timestamp} | {record.levelname:<7} | {scope:<20} | {record.getMessage()}{field_text}"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the `kube` namespace.

    This helper exists so callers can request `cli`, `dataset`, or other scopes
    without manually assembling dotted logger names.
    """
    if not name:
        return logging.getLogger("kube")
    return logging.getLogger(f"kube.{name}")


def configure_logging(
    *,
    level: str | int | None = None,
    quiet: bool = False,
    stream=None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure the shared `kube` logger tree for CLI-style usage.

    The function exists so command-line tools can emit status lines to stderr
    while leaving stdout free for JSON payloads and piped command output.
    """
    resolved_level = logging.ERROR if quiet else _coerce_level(level or os.getenv("KUBE_LOG_LEVEL", DEFAULT_LOG_LEVEL))
    root = get_logger()
    root.setLevel(resolved_level)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(stream or sys.stderr)
    console.setLevel(resolved_level)
    console.setFormatter(KubeFormatter())
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(KubeFormatter())
        root.addHandler(file_handler)

    return root


def log_event(logger: logging.Logger, level: int, message: str, /, **fields: Any) -> None:
    """Emit one structured Kube log line with optional key-value context.

    This wrapper exists because the standard logging API makes structured extras
    fairly verbose, and CLI handlers call this pattern repeatedly.
    """
    logger.log(level, message, extra={"kube_fields": fields})


__all__ = [
    "DEFAULT_LOG_LEVEL",
    "KubeFormatter",
    "LOG_LEVELS",
    "configure_logging",
    "get_logger",
    "log_event",
]
