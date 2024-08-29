from __future__ import annotations

import logging
import os
from datetime import date

from colorama import Fore, Style

SIMPLE_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
DEBUG_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)03d  %(message)s"
LOG_LEVEL = logging.DEBUG

def configure_logging(level: int = LOG_LEVEL) -> logging.Logger:
    """Configure the native logging module and return the logger."""
    
    # Auto-adjust default log format based on log level
    log_format = DEBUG_LOG_FORMAT if level == logging.DEBUG else SIMPLE_LOG_FORMAT

    # Ensure log directory exists
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Console handler with custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FancyConsoleFormatter(log_format))

    # File handler with standard formatter
    file_handler = logging.FileHandler(f"{log_dir}/{__name__}_{date.today()}.log")
    file_handler.setFormatter(logging.Formatter(log_format))

    # Add handlers to the logger
    if not logger.hasHandlers():  # To prevent adding handlers multiple times
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


class FancyConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter designed for console output.

    This formatter enhances the standard logging output with color coding. The color
    coding is based on the level of the log message, making it easier to distinguish
    between different types of messages in the console output.

    The color for each level is defined in the LEVEL_COLOR_MAP class attribute.
    """

    # level -> (text color)
    LEVEL_COLOR_MAP = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Ensure `msg` is a string
        if not isinstance(record.msg, str):
            record.msg = str(record.msg)

        # Justify the level name to 5 characters minimum
        record.levelname = record.levelname.ljust(5)

        # Determine default color based on error level
        level_color = self.LEVEL_COLOR_MAP.get(record.levelno, "")
        record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"

        # Determine color for message
        color = getattr(record, "color", level_color)
        color_is_specified = hasattr(record, "color")

        # Don't color INFO messages unless the color is explicitly specified
        if color and (record.levelno != logging.INFO or color_is_specified):
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

        return super().format(record)


def main():
    # Configure the logger
    logger = configure_logging()

    # Example logging
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    return logger


# Example usage:
if __name__ == "__main__":
    main()
