"""轻量日志工具。"""
from __future__ import annotations

import logging
import sys

_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "rm_arm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
