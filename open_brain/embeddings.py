"""
Embedding service for Open Brain memory system

Generates and manages vector embeddings using a local OpenAI-compatible
embedding server (e.g., LM Studio with Qwen3-Embedding-0.6B).
"""
import asyncio
import logging
import shutil
from typing import Optional

import httpx

from open_brain.config import EMBEDDING_API_URL, EMBEDDING_MODEL, LMSTUDIO_BASE_URL
from open_brain.database import KnowledgeBase

logger = logging.getLogger("open_brain.embed")


class EmbeddingService:
    """Handles embedding generation and storage via local embedding server."""

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()
        self.model = EMBEDDING_MODEL
        self.api_url = EMBEDDING_API_URL
        self._model_checked = False

    async def _ensure_model_loaded(self) -> None:
        """Check if embedding model is loaded in LM Studio; auto-load if not."""
        if self._model_checked:
            return
        self._model_checked = True

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{LMSTUDIO_BASE_URL}/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                loaded_ids = [m["id"] for m in models]

                if self.model in loaded_ids:
                    logger.info("Embedding model '%s' already loaded", self.model)
                    return

                logger.info("Embedding model '%s' not loaded — attempting auto-load", self.model)
        except httpx.ConnectError:
            logger.warning("LM Studio not reachable at %s — skipping auto-load", LMSTUDIO_BASE_URL)
            return
        except Exception as e:
            logger.warning("Could not check LM Studio models: %s", e)
            return

        # Try to load via lms CLI
        lms_path = shutil.which("lms")
        if not lms_path:
            logger.warning("'lms' CLI not found in PATH — cannot auto-load model")
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                lms_path, "load", self.model, "--yes",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                logger.warning("lms load failed (exit %d): %s", proc.returncode, stderr.decode().strip())
                return
            logger.info("lms load started for '%s'", self.model)
        except asyncio.TimeoutError:
            logger.warning("lms load timed out after 30s")
            return
        except Exception as e:
            logger.warning("Failed to run lms load: %s", e)
            return

        # Poll /v1/models until model appears (up to 15s)
        for _ in range(15):
            await asyncio.sleep(1)
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{LMSTUDIO_BASE_URL}/v1/models")
                    resp.raise_for_status()
                    models = resp.json().get("data", [])
                    if self.model in [m["id"] for m in models]:
                        logger.info("Embedding model '%s' loaded successfully", self.model)
                        return
            except Exception:
                pass

        logger.warning("Embedding model '%s' did not appear after 15s — embeddings may fail", self.model)

    async def generate_embedding(self, text: str) -> Optional[list[float]]:
        """Generate embedding using local OpenAI-compatible server."""
        if not self.api_url:
            logger.debug("EMBEDDING_API_URL not set - skipping embeddings")
            return None

        await self._ensure_model_loaded()

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "input": text[:8000],
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                return data["data"][0]["embedding"]
            except httpx.ConnectError:
                logger.warning("Embedding server unreachable at %s", self.api_url)
                return None
            except httpx.HTTPStatusError as e:
                logger.error("Embedding API error: %s - %s", e.response.status_code, e.response.text[:200])
                return None
            except Exception as e:
                logger.error("Embedding failed: %s", e)
                return None

    async def chunk_content(self, content: str, max_length: int = 500) -> list[str]:
        """Split long content into chunks for embedding."""
        chunks = []
        current_chunk = []
        current_length = 0

        lines = content.split('\n')

        for line in lines:
            line_length = len(line) + 1

            if current_length + line_length > max_length:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        # Split oversized chunks by words
        result = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                result.append(chunk)
            else:
                words = chunk.split()
                current = []
                for word in words:
                    if len(' '.join(current + [word])) > max_length:
                        if current:
                            result.append(' '.join(current))
                        current = [word]
                    else:
                        current.append(word)
                if current:
                    result.append(' '.join(current))

        return result

    async def process_message(self, message: dict) -> Optional[str]:
        """Full pipeline: store message -> chunk -> embed -> store embeddings."""
        msg_id = await self.kb.store_message(message)

        if not msg_id:
            logger.error("Failed to store message in brain")
            return None

        chunks = await self.chunk_content(message["content"])

        if not chunks:
            return msg_id

        for i, chunk in enumerate(chunks):
            embedding = await self.generate_embedding(chunk)

            if embedding:
                await self.kb.store_embedding(
                    message_id=msg_id,
                    content=chunk,
                    embedding=embedding,
                    metadata={
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_length": len(chunk)
                    }
                )

        return msg_id

    async def search_memory(self, query: str, limit: int = 5) -> list[dict]:
        """Search memory using semantic similarity."""
        query_embedding = await self.generate_embedding(query)

        if not query_embedding:
            return []

        return await self.kb.semantic_search(query, query_embedding, limit)

    async def get_embeddings_for_message(self, msg_id: str) -> list[dict]:
        """Get all embeddings for a specific message."""
        return await self.kb.get_embeddings_by_message(msg_id)


# --- Singleton convenience functions ---

_service: Optional[EmbeddingService] = None


def _get_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


async def process_message_async(message: dict) -> Optional[str]:
    return await _get_service().process_message(message)


async def search_memory_async(query: str, limit: int = 5) -> list[dict]:
    return await _get_service().search_memory(query, limit)
