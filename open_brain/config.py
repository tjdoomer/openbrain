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

# Embeddings (LM Studio / OpenAI-compatible server)
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b-mxfp8")

# LM Studio base URL (derived from EMBEDDING_API_URL, or override directly)
LMSTUDIO_BASE_URL = os.getenv(
    "LMSTUDIO_BASE_URL",
    EMBEDDING_API_URL.rsplit("/v1/", 1)[0] if "/v1/" in EMBEDDING_API_URL else "http://localhost:1234"
)

# Obsidian
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT", "")
OBSIDIAN_SUMMARY_DIR = os.path.join(OBSIDIAN_VAULT_PATH, "BuddyChat")
OBSIDIAN_TEMPLATE = Path(os.getenv("OBSIDIAN_TEMPLATE", ""))
SUMMARY_INTERVAL_HOURS = int(os.getenv("SUMMARY_INTERVAL_HOURS", "2"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
