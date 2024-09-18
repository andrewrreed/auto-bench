from loguru import logger
import sys


def setup_logging():
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    logger.add(
        "./logs/autobench.log", rotation="10 MB", retention="1 week", level="DEBUG"
    )
