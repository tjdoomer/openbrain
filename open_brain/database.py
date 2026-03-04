"""
Open Brain database layer with PostgreSQL + PGVector support

Supports both SQLite (development) and PostgreSQL with PGVector extension (production).
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

try:
    from sqlalchemy import create_engine, text, Column, String, Text, DateTime, JSON
    from sqlalchemy.orm import sessionmaker, declarative_base
except ImportError:
    raise ImportError("Install SQLAlchemy: pip install sqlalchemy")

from open_brain.config import (
    DATABASE_URL, USE_POSTGRES,
    PGVECTOR_HOST, PGVECTOR_PORT, PGVECTOR_DB,
    PGVECTOR_USER, PGVECTOR_PASSWORD
)

logger = logging.getLogger("open_brain.db")

# Choose column types based on backend
if USE_POSTGRES:
    from sqlalchemy.dialects.postgresql import JSONB as JSONType
    from pgvector.sqlalchemy import Vector
    EMBEDDING_DIM = 768  # Qwen3-Embedding-0.6B
    EmbeddingColumn = Vector(EMBEDDING_DIM)
else:
    JSONType = JSON
    EmbeddingColumn = Text  # JSON array of floats (SQLite fallback)

Base = declarative_base()


class Message(Base):
    """Chat message stored in database"""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True)
    room = Column(String(50), nullable=False, index=True)
    sender_id = Column(String(50), nullable=False)
    sender_name = Column(String(100), nullable=False)
    sender_type = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    meta = Column("metadata", JSONType, nullable=True)
    created_at = Column(DateTime, nullable=False, index=True)


class Embedding(Base):
    """Vector embeddings for semantic search"""
    __tablename__ = "embeddings"

    id = Column(String(36), primary_key=True)
    message_id = Column(String(36), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(EmbeddingColumn)
    meta = Column("metadata", JSONType, nullable=True)
    created_at = Column(DateTime, nullable=False)


class KnowledgeBase:
    """Manages both message storage and vector embeddings"""

    def __init__(self):
        if USE_POSTGRES:
            self._setup_postgres()
        else:
            self._setup_sqlite()

    def _setup_postgres(self):
        """Configure PostgreSQL with PGVector extension"""
        url = f"postgresql+psycopg2://{PGVECTOR_USER}:{PGVECTOR_PASSWORD}@{PGVECTOR_HOST}:{PGVECTOR_PORT}/{PGVECTOR_DB}"
        self.engine = create_engine(url, echo=False)

        with self.engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception as e:
                logger.warning("Could not enable vector extension: %s", e)

        Base.metadata.create_all(self.engine)

        # Create vector similarity index (IVFFlat) if it doesn't exist
        with self.engine.connect() as conn:
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS ix_embeddings_vector
                    ON embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))
                conn.commit()
                logger.info("PGVector index ready (ivfflat, cosine, lists=100)")
            except Exception as e:
                # IVFFlat requires data to build — will succeed after migration
                logger.info("Vector index deferred (needs data first): %s", e)

        self.Session = sessionmaker(bind=self.engine)

    def _setup_sqlite(self):
        """Configure SQLite for development"""
        db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        """Get a database session"""
        return self.Session()

    # --- Async wrappers (run sync SQLAlchemy in thread) ---

    async def store_message(self, message_data: dict) -> str:
        return await asyncio.to_thread(self._store_message_sync, message_data)

    async def store_embedding(self, message_id: str, content: str,
                             embedding: list[float], metadata: dict = None):
        return await asyncio.to_thread(
            self._store_embedding_sync, message_id, content, embedding, metadata
        )

    async def semantic_search(self, query: str, embedding: list[float],
                             limit: int = 5) -> list[dict]:
        return await asyncio.to_thread(self._semantic_search_sync, query, embedding, limit)

    async def get_recent_messages(self, room: str = "general", limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._get_recent_messages_sync, room, limit)

    async def get_message_by_id(self, msg_id: str) -> Optional[dict]:
        return await asyncio.to_thread(self._get_message_by_id_sync, msg_id)

    async def get_embeddings_by_message(self, msg_id: str) -> list[dict]:
        return await asyncio.to_thread(self._get_embeddings_by_message_sync, msg_id)

    async def find_note_by_path(self, file_path: str) -> Optional[dict]:
        """Find an Obsidian note by its file path in metadata."""
        return await asyncio.to_thread(self._find_note_by_path_sync, file_path)

    async def upsert_note(self, note_data: dict) -> str:
        """Insert or update an Obsidian note by file_path."""
        return await asyncio.to_thread(self._upsert_note_sync, note_data)

    async def delete_embeddings_for_message(self, msg_id: str):
        """Delete all embeddings for a message (for re-embedding)."""
        return await asyncio.to_thread(self._delete_embeddings_for_message_sync, msg_id)

    # --- Sync implementations ---

    def _store_message_sync(self, message_data: dict) -> str:
        msg_id = str(uuid4())
        session = self.get_session()
        try:
            message = Message(
                id=msg_id,
                room=message_data["room"],
                sender_id=message_data["sender_id"],
                sender_name=message_data["sender_name"],
                sender_type=message_data["sender_type"],
                content=message_data["content"],
                meta=message_data.get("metadata"),
                created_at=message_data.get("created_at", datetime.now(timezone.utc))
            )
            session.add(message)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
        return msg_id

    def _store_embedding_sync(self, message_id: str, content: str,
                              embedding: list[float], metadata: dict = None):
        session = self.get_session()
        try:
            emb = Embedding(
                id=str(uuid4()),
                message_id=message_id,
                content=content,
                embedding=embedding if USE_POSTGRES else json.dumps(embedding),
                meta=metadata,
                created_at=datetime.now(timezone.utc)
            )
            session.add(emb)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _semantic_search_sync(self, query: str, embedding: list[float],
                              limit: int = 5) -> list[dict]:
        session = self.get_session()
        try:
            if USE_POSTGRES:
                results = session.execute(
                    text("""
                        SELECT message_id, content, metadata,
                               1 - (embedding <=> CAST(:query_vec AS vector)) as similarity
                        FROM embeddings
                        WHERE 1 - (embedding <=> CAST(:query_vec AS vector)) > 0.5
                        ORDER BY similarity DESC
                        LIMIT :limit
                    """),
                    {"query_vec": str(embedding), "limit": limit}
                ).fetchall()
            else:
                all_embeddings = session.query(Embedding).all()

                scored = []
                for emb in all_embeddings:
                    try:
                        emb_vec = json.loads(emb.embedding)
                        similarity = self._cosine_similarity(embedding, emb_vec)
                        if similarity > 0.5:
                            scored.append((emb, similarity))
                    except (json.JSONDecodeError, ValueError):
                        continue

                scored.sort(key=lambda x: x[1], reverse=True)
                results = [(e.message_id, e.content, e.meta, s)
                          for e, s in scored[:limit]]

            return [
                {
                    "message_id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2],
                    "similarity": float(row[3])
                }
                for row in results
            ]
        finally:
            session.close()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _get_recent_messages_sync(self, room: str = "general", limit: int = 50) -> list[dict]:
        session = self.get_session()
        try:
            results = session.query(Message)\
                .filter_by(room=room)\
                .order_by(Message.created_at.desc())\
                .limit(limit)\
                .all()

            messages = [
                {
                    "id": msg.id,
                    "room": msg.room,
                    "sender_id": msg.sender_id,
                    "sender_name": msg.sender_name,
                    "sender_type": msg.sender_type,
                    "content": msg.content,
                    "metadata": msg.meta,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None
                }
                for msg in results
            ]
            messages.reverse()
            return messages
        finally:
            session.close()

    def _get_message_by_id_sync(self, msg_id: str) -> Optional[dict]:
        session = self.get_session()
        try:
            msg = session.query(Message).filter_by(id=msg_id).first()
            if not msg:
                return None
            return {
                "id": msg.id,
                "room": msg.room,
                "sender_id": msg.sender_id,
                "sender_name": msg.sender_name,
                "sender_type": msg.sender_type,
                "content": msg.content,
                "metadata": msg.meta,
                "created_at": msg.created_at.isoformat() if msg.created_at else None
            }
        finally:
            session.close()

    def _get_embeddings_by_message_sync(self, msg_id: str) -> list[dict]:
        session = self.get_session()
        try:
            embeddings = session.query(Embedding)\
                .filter_by(message_id=msg_id)\
                .all()
            return [
                {
                    "id": emb.id,
                    "message_id": emb.message_id,
                    "content": emb.content,
                    "metadata": emb.meta,
                    "created_at": emb.created_at.isoformat() if emb.created_at else None
                }
                for emb in embeddings
            ]
        finally:
            session.close()

    def _find_note_by_path_sync(self, file_path: str) -> Optional[dict]:
        """Find a message with source=obsidian and matching file_path in metadata."""
        session = self.get_session()
        try:
            results = session.query(Message)\
                .filter_by(room="obsidian", sender_type="system")\
                .all()
            for msg in results:
                meta = msg.meta if isinstance(msg.meta, dict) else {}
                if meta.get("file_path") == file_path:
                    return {
                        "id": msg.id,
                        "content": msg.content,
                        "metadata": meta,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    }
            return None
        finally:
            session.close()

    def _upsert_note_sync(self, note_data: dict) -> str:
        """Insert or update an Obsidian note. Returns message ID."""
        file_path = note_data["file_path"]
        existing = self._find_note_by_path_sync(file_path)

        session = self.get_session()
        try:
            if existing:
                msg = session.query(Message).filter_by(id=existing["id"]).first()
                msg.content = note_data["content"]
                msg.meta = {
                    "source": "obsidian",
                    "file_path": file_path,
                    "folder": note_data.get("folder", ""),
                    "file_modified": note_data.get("file_modified"),
                }
                msg.created_at = datetime.now(timezone.utc)
                session.commit()
                return existing["id"]
            else:
                msg_id = str(uuid4())
                message = Message(
                    id=msg_id,
                    room="obsidian",
                    sender_id="obsidian-vault",
                    sender_name="Obsidian",
                    sender_type="system",
                    content=note_data["content"],
                    meta={
                        "source": "obsidian",
                        "file_path": file_path,
                        "folder": note_data.get("folder", ""),
                        "file_modified": note_data.get("file_modified"),
                    },
                    created_at=datetime.now(timezone.utc),
                )
                session.add(message)
                session.commit()
                return msg_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _delete_embeddings_for_message_sync(self, msg_id: str):
        """Delete all embeddings for a message."""
        session = self.get_session()
        try:
            session.query(Embedding).filter_by(message_id=msg_id).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
