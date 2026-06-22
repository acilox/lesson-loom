"""structlog setup — quiet by default, human-readable when LESSON_LOOM_LOG=1."""

from __future__ import annotations

import logging
import os

import structlog


def configure_logging() -> None:
    enabled = os.getenv("LESSON_LOOM_LOG", "0") == "1"
    level = logging.INFO if enabled else logging.WARNING
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=enabled),
        ],
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "lesson_loom"):
    return structlog.get_logger(name)
