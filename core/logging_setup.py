"""
Centralised logging setup.

Call setup_logging() once at startup (main.py).
Every module then just does:  logger = logging.getLogger(__name__)

Log files:
  logs/app.log        — all INFO+ messages, rotated at 5 MB, 5 backups
  logs/agent.log      — agent / LangGraph only (DEBUG+), for deep inspection
  logs/errors.log     — ERROR+ only, never rotated away (append-only)

Console output mirrors logs/app.log at the configured level.
"""

from __future__ import annotations
import logging
import logging.handlers
import os
import sys


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
) -> None:
    os.makedirs(log_dir, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # root catches everything; handlers filter

    # ── Console ────────────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── app.log — general rotating log ────────────────────────────────────
    app_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    app_handler.setLevel(numeric_level)
    app_handler.setFormatter(fmt)
    root.addHandler(app_handler)

    # ── agent.log — LangGraph / agent DEBUG detail ─────────────────────────
    agent_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "agent.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    agent_handler.setLevel(logging.DEBUG)
    agent_handler.setFormatter(fmt)
    agent_handler.addFilter(_PrefixFilter(("agent.", "langgraph", "langchain")))
    root.addHandler(agent_handler)

    # ── errors.log — ERROR+ never rotated, always appended ────────────────
    error_handler = logging.FileHandler(
        filename=os.path.join(log_dir, "errors.log"),
        mode="a",
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)

    logging.getLogger(__name__).info(
        "Logging initialised | level=%s | dir=%s", log_level, log_dir
    )


class _PrefixFilter(logging.Filter):
    """Pass only records whose logger name starts with one of the given prefixes."""

    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name.startswith(p) for p in self._prefixes)
