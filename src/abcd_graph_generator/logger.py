__all__ = ["get_logger"]

import logging
import sys

LOGGER_NAME = "ABCDGraphGenerator"
LOG_FORMAT = "[%(name)s] [%(levelname)s] %(message)s"


class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.levelname = record.levelname.lower().ljust(len("warning"))
        formatter = logging.Formatter(LOG_FORMAT)
        return formatter.format(record)


class StdoutHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        logging.Handler.__init__(self, level)

    @property
    def stream(self):
        return sys.stdout


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def _set_up_logging():
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    stdout_handler = StdoutHandler()
    stdout_handler.setFormatter(CustomFormatter())
    logger.addHandler(stdout_handler)


_set_up_logging()
