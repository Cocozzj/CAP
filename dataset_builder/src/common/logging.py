from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: str = "INFO") -> Any:
    """Configure structured logging when available, with a stdlib fallback."""
    try:
        from rich.logging import RichHandler

        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except ImportError:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
        )
        return structlog.get_logger()
    except ImportError:
        return logging.getLogger("dataset_builder")
