"""
Open Brain — self-contained configuration

All settings via environment variables. No external dependencies.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Server
BRAIN_HOST = os.getenv("BRAIN_HOST", "0.0.0.0")
BRAIN_PORT = int(os.getenv("BRAIN_PORT", "8766"))

# Database
USE_POSTGRES = os.getenv("USE_POSTGRES", "true").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/brain.db")

# PGVector
PGVECTOR_HOST = os.getenv("PGVECTOR_HOST", "localhost")
PGVECTOR_PORT = int(os.getenv("PGVECTOR_PORT", "5432"))
PGVECTOR_DB = os.getenv("PGVECTOR_DB", "braindb")
PGVECTOR_USER = os.getenv("PGVECTOR_USER", "brainuser")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "")

# Embeddings (local MLX model — Qwen3-Embedding-0.6B)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b")
EMBEDDING_MODEL_PATH = Path(os.getenv(
    "EMBEDDING_MODEL_PATH",
    str(Path(__file__).parent.parent / "models" / "qwen3-embedding-0.6b"),
))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Obsidian
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT", "")
OBSIDIAN_SUMMARY_DIR = os.path.join(OBSIDIAN_VAULT_PATH, "BuddyChat") if OBSIDIAN_VAULT_PATH else ""
OBSIDIAN_TEMPLATE = Path(os.getenv("OBSIDIAN_TEMPLATE", ""))
SUMMARY_INTERVAL_HOURS = int(os.getenv("SUMMARY_INTERVAL_HOURS", "2"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
