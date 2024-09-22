from loguru import logger
import sys


def setup_logging():

    LOG_LEVEL = "SUCCESS" if "ipykernel" in sys.modules else "INFO"
    LOG_LEVEL = "INFO"

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
    )
    logger.add(
        "./logs/autobench.log", rotation="10 MB", retention="1 week", level="DEBUG"
    )
