"""Centralised logging.

A tiny wrapper on top of the standard library so that every module can call
`get_logger(__name__)` without re-configuring handlers. The first call sets
up a uniform formatter; subsequent calls return cached loggers.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def _configure_root(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    The level is taken from ``level`` arg, then ``LOG_LEVEL`` env var, then INFO.
    """
    effective = level or os.getenv("LOG_LEVEL", "INFO")
    _configure_root(effective)
    logger = logging.getLogger(name)
    logger.setLevel(effective.upper())
    return logger
