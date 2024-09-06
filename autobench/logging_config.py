from loguru import logger
import sys


def setup_logging():
    # Remove any default handlers
    logger.remove()

    # Add a handler to write to stderr with a specific format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add a handler to write to a file
    logger.add(
        "logs/autobench.log", rotation="10 MB", retention="1 week", level="DEBUG"
    )
