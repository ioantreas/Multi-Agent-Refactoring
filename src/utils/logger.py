import logging
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

log_file = (
        LOG_DIR /
        f"run_{timestamp}.log"
)

DEBUG = True


def setup_logger():

    logger = logging.getLogger(
        "refactor-agent"
    )

    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(message)s"
    )

    # Console output
    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(
        logging.DEBUG
        if DEBUG
        else logging.INFO
    )

    console_handler.setFormatter(
        formatter
    )

    # File logging
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.DEBUG
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    logger.propagate = False

    logger.info(
        f"[Logging to] {log_file}"
    )

    return logger


logger = setup_logger()
