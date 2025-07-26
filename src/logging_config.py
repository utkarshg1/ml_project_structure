import os
import sys
from loguru import logger

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logger.remove()

# Console (stream) handler
logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")

# File handler
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    enqueue=True,
)
