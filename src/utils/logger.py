"""
Centralized logger setup.

Why not just use print()? In a production system, print statements can't be
turned off, filtered by severity, or redirected to a file/monitoring system.
Python's `logging` module gives us all of that for free.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Create (or fetch) a logger with a consistent format.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Detector loaded successfully")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers if called multiple times
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
