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


_ANSI_RESET = "\x1b[0m"
_LEVEL_COLOR = {
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[31m",  # red
}


def _color_enabled(stream) -> bool:
    """Decide whether ANSI colors should be written to `stream`.

    Exists so colored output only reaches real terminals: piped/redirected
    output and log files stay plain text, and `NO_COLOR` is honored per the
    https://no-color.org convention.
    """
    if os.getenv("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


class KubeFormatter(logging.Formatter):
    """Format log records into a terse, terminal-native Kube line.

    The formatter exists to keep Kube's CLI output reading like a small tool's
    console chatter rather than a structured/API access log — a short tag, a
    plain message, and bare values trailing after it with no `key=` noise.
    Warnings and errors are colored (yellow/red) when writing to a real
    terminal, so they stand out without needing extra words.
    """

    def __init__(self, *, color: bool = False) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as `[scope] message  values...`."""
        scope = record.name
        if scope.startswith("kube."):
            scope = scope[5:]
        prefix = f"[{scope}]"
        fields = getattr(record, "kube_fields", None) or {}
        parts = []
        for key, value in fields.items():
            if value is None:
                continue
            text = _stringify_field(value)
            # Self-evident values (paths, shapes, dtypes) read fine bare; short
            # scalars (counts, flags, modes) need their key or they're noise.
            if len(text) <= 6 and not ("x" in text and text[0].isdigit()):
                parts.append(f"{key}={text}")
            else:
                parts.append(text)
        field_text = "  " + "  ".join(parts) if parts else ""
        line = f"{prefix} {record.getMessage()}{field_text}"

        color = self._color and _LEVEL_COLOR.get(record.levelno)
        if color:
            line = f"{color}{line}{_ANSI_RESET}"
        return line


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

    console_stream = stream or sys.stderr
    console = logging.StreamHandler(console_stream)
    console.setLevel(resolved_level)
    console.setFormatter(KubeFormatter(color=_color_enabled(console_stream)))
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(KubeFormatter(color=False))
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
