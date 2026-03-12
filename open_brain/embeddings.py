"""
Embedding service for Open Brain memory system

Generates and manages vector embeddings using a local MLX model
(Qwen3-Embedding-0.6B, 1024-dim, Apple Silicon optimized).
"""
import asyncio
import logging
from typing import Optional

from open_brain.config import EMBEDDING_MODEL, EMBEDDING_MODEL_PATH
from open_brain.database import KnowledgeBase

logger = logging.getLogger("open_brain.embed")

# Qwen3 uses an instruction prefix for asymmetric retrieval (queries vs documents)
QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

# --- Lazy singleton model ---

_model = None
_tokenizer = None


def _get_model():
    """Load the MLX embedding model once (thread-safe via GIL)."""
    global _model, _tokenizer
    if _model is None:
        import mlx_embeddings
        model_path = str(EMBEDDING_MODEL_PATH)
        logger.info("Loading MLX embedding model '%s' from %s", EMBEDDING_MODEL, model_path)
        _model, _tokenizer = mlx_embeddings.load(model_path)
        logger.info("MLX embedding model loaded")
    return _model, _tokenizer


class EmbeddingService:
    """Handles embedding generation and storage via local MLX model."""

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()
        self.model_name = EMBEDDING_MODEL

    async def generate_embedding(
        self, text: str, is_query: bool = False
    ) -> Optional[list[float]]:
        """Generate embedding using local MLX model.

        Args:
            text: Text to embed (truncated to 8000 chars).
            is_query: If True, prepend the Qwen3 query instruction prefix.
                      Use True for search queries, False for documents/content.
        """
        try:
            def _embed():
                import mlx_embeddings
                model, tokenizer = _get_model()
                input_text = text[:8000]
                if is_query:
                    input_text = QUERY_PREFIX + input_text
                result = mlx_embeddings.generate(model, tokenizer, input_text)
                return result.text_embeds[0].tolist()

            return await asyncio.to_thread(_embed)
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
        query_embedding = await self.generate_embedding(query, is_query=True)

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
