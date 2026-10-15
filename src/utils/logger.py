"""
logger.py

Centralized logging configuration for HealthAudit-AI.

Every module in this project should obtain its logger via `get_logger(__name__)`
instead of calling `print()` or configuring `logging` directly. This guarantees:
    - Consistent log formatting across the entire codebase.
    - Every log line is traceable to the exact module that produced it.
    - Logs are persisted to a file (outputs/logs/) in addition to the console,
      which matters once this code runs unattended (e.g. behind FastAPI).
"""

import logging
import sys
from logging import Logger

from src.config.paths import LOGS_DIR

# Log message format:
#   2024-05-01 10:15:32 | INFO | src.preprocessing.data_loader | Loaded 116352 rows
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_FILE_PATH = LOGS_DIR / "healthaudit_ai.log"


def get_logger(name: str, level: int = logging.INFO) -> Logger:
    """
    Return a configured logger for the given module name.

    Parameters
    ----------
    name : str
        Typically passed as `__name__` from the calling module, so log
        lines are automatically tagged with their source (e.g.
        'src.preprocessing.data_loader').
    level : int
        Minimum severity level to log. Defaults to logging.INFO.
        Use logging.DEBUG during development for more verbose output.

    Returns
    -------
    logging.Logger
        A logger that writes to both the console and a shared log file,
        with duplicate handlers prevented on repeated calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Guard: if this logger already has handlers attached (e.g. get_logger
    # was called earlier for the same module), don't attach them again --
    # otherwise every log message would be printed multiple times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler: for real-time feedback while developing/running scripts.
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler: persists logs to disk so they survive after the process
    # ends -- essential once this runs as a backend service, not just a script.
    file_handler = logging.FileHandler(filename=LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent messages from propagating to the root logger, which could
    # otherwise cause duplicate output if the root logger also has handlers.
    logger.propagate = False

    return logger


if __name__ == "__main__":
    # Running this file directly (`python src/utils/logger.py`) lets us
    # sanity-check the logging setup in isolation, without needing any
    # other part of the pipeline to exist yet.
    demo_logger = get_logger(__name__)
    demo_logger.debug("This DEBUG message will NOT appear (level is INFO).")
    demo_logger.info("Logger initialized successfully.")
    demo_logger.warning("This is a sample warning message.")
    demo_logger.error("This is a sample error message.")
    print(f"\nLog file written to: {LOG_FILE_PATH}")