import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Export the configured logger
__all__ = ['logger'] 