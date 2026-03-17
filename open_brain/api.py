"""
Open Brain — FastAPI REST API

Standalone brain service exposing semantic search, message capture,
Obsidian sync, and health endpoints.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from open_brain.config import BRAIN_HOST, BRAIN_PORT, OBSIDIAN_VAULT_PATH
from open_brain.database import KnowledgeBase
from open_brain.embeddings import EmbeddingService

logger = logging.getLogger("open_brain.api")

# --- Singletons ---

_kb: Optional[KnowledgeBase] = None
_embed: Optional[EmbeddingService] = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def get_embed() -> EmbeddingService:
    global _embed
    if _embed is None:
        _embed = EmbeddingService(kb=get_kb())
    return _embed


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_kb()
    logger.info("Open Brain started — database connected")
    yield
    logger.info("Open Brain shutting down")


# --- App ---

app = FastAPI(title="Open Brain", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---

class CaptureRequest(BaseModel):
    room: str = "general"
    sender_id: str
    sender_name: str
    sender_type: str
    content: str
    metadata: Optional[dict] = None


class NoteUpsertRequest(BaseModel):
    file_path: str
    folder: str = ""
    content: str
    file_modified: Optional[float] = None


class TaskCreateRequest(BaseModel):
    summary: str
    description: Optional[str] = None
    status: str = "open"
    priority: Optional[str] = None
    project: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict] = None


class TaskUpdateRequest(BaseModel):
    summary: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    project: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict] = None


# --- Health ---

@app.get("/health")
async def health():
    """Check brain service, database, and embedding server status."""
    status = {"brain": "ok", "database": "unknown", "embeddings": "unknown"}

    # Check database
    try:
        kb = get_kb()
        await kb.get_recent_messages("general", limit=1)
        status["database"] = "ok"
    except Exception as e:
        status["database"] = f"error: {e}"

    # Check local MLX embedding model
    try:
        from open_brain.embeddings import _get_model
        _get_model()
        status["embeddings"] = "ok (mlx-qwen3)"
    except Exception as e:
        status["embeddings"] = f"unavailable: {e}"

    overall = "ok" if status["database"] == "ok" else "degraded"
    return {"status": overall, "components": status}


# --- Capture (write) ---

@app.post("/api/capture")
async def capture_message(body: CaptureRequest):
    """Store + chunk + embed a message. Fire-and-forget friendly."""
    try:
        embed = get_embed()
        msg_id = await embed.process_message({
            "room": body.room,
            "sender_id": body.sender_id,
            "sender_name": body.sender_name,
            "sender_type": body.sender_type,
            "content": body.content,
            "metadata": body.metadata,
        })
        return {"message_id": msg_id, "status": "captured"}
    except Exception as e:
        logger.error("Capture failed: %s", e)
        raise HTTPException(500, f"Capture failed: {e}")


# --- Search (read) ---

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=50, description="Max results"),
):
    """Semantic vector search across all stored content."""
    embed = get_embed()
    results = await embed.search_memory(q, limit=limit)
    return results


# --- Recent messages (read) ---

@app.get("/api/recent")
async def recent(
    room: str = Query("general", description="Room name"),
    limit: int = Query(20, ge=1, le=200, description="Number of messages"),
):
    """Get recent messages from the brain database."""
    kb = get_kb()
    messages = await kb.get_recent_messages(room, limit=limit)
    return messages


# --- Context (read) ---

@app.get("/api/context")
async def context(
    topic: str = Query("", description="Topic to find context for"),
    message_id: str = Query("", description="Specific message ID"),
    before: int = Query(3, ge=0, le=20, description="Messages before"),
    after: int = Query(3, ge=0, le=20, description="Messages after"),
):
    """Get conversation context around a topic or message."""
    kb = get_kb()
    embed = get_embed()

    if not topic and not message_id:
        raise HTTPException(400, "Provide 'topic' or 'message_id'")

    if message_id:
        anchor = await kb.get_message_by_id(message_id)
        if not anchor:
            raise HTTPException(404, f"Message {message_id} not found")
        room = anchor["room"]
    else:
        embedding = await embed.generate_embedding(topic, is_query=True)
        if not embedding:
            raise HTTPException(503, "Could not generate embedding")
        results = await kb.semantic_search(topic, embedding, 1)
        if not results:
            return {"messages": [], "anchor": None, "note": f"No results for: {topic}"}
        anchor = await kb.get_message_by_id(results[0]["message_id"])
        if not anchor:
            raise HTTPException(404, "Anchor message not found")
        room = anchor["room"]

    all_msgs = await kb.get_recent_messages(room, limit=200)

    anchor_idx = None
    anchor_id = message_id or anchor["id"]
    for i, m in enumerate(all_msgs):
        if m["id"] == anchor_id:
            anchor_idx = i
            break

    if anchor_idx is None:
        return {"messages": [], "anchor": anchor, "note": "Could not locate in timeline"}

    start = max(0, anchor_idx - before)
    end = min(len(all_msgs), anchor_idx + after + 1)
    context_msgs = all_msgs[start:end]

    return {
        "messages": context_msgs,
        "anchor": anchor,
        "anchor_index": anchor_idx - start,
    }


# --- Notes (Obsidian) ---

@app.post("/api/notes/upsert")
async def upsert_note(body: NoteUpsertRequest):
    """Upsert an Obsidian note into the brain database and write to vault."""
    kb = get_kb()
    embed = get_embed()

    msg_id = await kb.upsert_note({
        "file_path": body.file_path,
        "folder": body.folder,
        "content": body.content,
        "file_modified": body.file_modified,
    })

    # Write .md file to Obsidian vault if configured
    vault_written = False
    if OBSIDIAN_VAULT_PATH:
        try:
            vault_file = Path(OBSIDIAN_VAULT_PATH) / body.file_path
            await asyncio.to_thread(_write_vault_file, vault_file, body.content)
            vault_written = True
            logger.info("Wrote note to vault: %s", vault_file)
        except Exception as e:
            logger.error("Failed to write note to vault: %s", e)

    # Re-embed the note
    await kb.delete_embeddings_for_message(msg_id)
    chunks = await embed.chunk_content(body.content, max_length=500)
    embedded_count = 0
    for i, chunk in enumerate(chunks):
        embedding = await embed.generate_embedding(chunk)
        if embedding:
            await kb.store_embedding(
                message_id=msg_id,
                content=chunk,
                embedding=embedding,
                metadata={
                    "source": "obsidian",
                    "file_path": body.file_path,
                    "folder": body.folder,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            embedded_count += 1

    return {
        "message_id": msg_id,
        "chunks": len(chunks),
        "embedded": embedded_count,
        "vault_written": vault_written,
    }


def _write_vault_file(path: Path, content: str) -> None:
    """Write content to a file in the Obsidian vault, creating dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# --- Tasks ---

@app.post("/api/tasks")
async def create_task(body: TaskCreateRequest):
    """Create a new task, embed it for semantic search."""
    kb = get_kb()
    embed = get_embed()

    task = await kb.create_task(body.model_dump(exclude_none=True))

    embedded = await embed.embed_task(
        task["id"], task["summary"], task.get("description") or ""
    )
    task["embedded_chunks"] = embedded
    return task


@app.get("/api/tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter: open|in_progress|done|blocked|all"),
    project: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
):
    """List tasks. Default: active (non-done) tasks."""
    kb = get_kb()
    return await kb.list_tasks(status=status, project=project, limit=limit)


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a task by short_id (TASK-1) or UUID."""
    kb = get_kb()
    task = await kb.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    return task


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: str, body: TaskUpdateRequest):
    """Partial update a task. Re-embeds if summary or description changed."""
    kb = get_kb()
    embed = get_embed()

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")

    task = await kb.update_task(task_id, updates)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")

    if "summary" in updates or "description" in updates:
        embedded = await embed.embed_task(
            task["id"], task["summary"], task.get("description") or ""
        )
        task["embedded_chunks"] = embedded

    return task


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its embeddings."""
    kb = get_kb()
    deleted = await kb.delete_task(task_id)
    if not deleted:
        raise HTTPException(404, f"Task {task_id} not found")
    return {"deleted": True, "task_id": task_id}


@app.post("/api/obsidian/sync")
async def obsidian_sync(force_full: bool = False):
    """Trigger a full Obsidian vault sync."""
    from open_brain.obsidian import import_vault
    stats = await import_vault(force_full=force_full)
    return stats


# --- Run ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BRAIN_HOST, port=BRAIN_PORT)
