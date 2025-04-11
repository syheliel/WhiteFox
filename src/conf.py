from pathlib import Path

from loguru import logger
import sys
# Remove default handler
logger.remove()

# Add custom handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
TORCH_BASE = Path("/home/synsyn/pytorch").resolve()
CHROMADB_PATH = (Path(__file__).parent.parent / "vectordb" ).resolve()
CHROMADB_PATH.mkdir(parents=True, exist_ok=True)

CHROMADB_COLLECTION = "torch"

OPTINFO_PATH = Path(__file__).parent.parent / "optim-0" / "optinfo.yaml"