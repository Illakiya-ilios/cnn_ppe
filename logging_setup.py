"""
Centralized logging configuration.

Provides a single get_logger() used across the app so output is consistent,
timestamped, levelled, and written both to the console and a rotating file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import config

_CONFIGURED = False


def configure():
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(os.path.dirname(config.LOG_FILE) or ".", exist_ok=True)

    level = getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("ppe")
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    fileh = RotatingFileHandler(
        config.LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fileh.setFormatter(fmt)
    root.addHandler(fileh)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name):
    configure()
    return logging.getLogger(f"ppe.{name}")
