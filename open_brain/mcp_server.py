"""
Open Brain — Standalone MCP Server

Exposes brain tools (semantic search, recent messages, context retrieval)
as MCP tools for Claude Desktop / Claude Code.

Usage in claude_desktop_config.json or .claude.json:
{
  "mcpServers": {
    "open-brain": {
      "command": "python",
      "args": ["~/open-brain/open_brain/mcp_server.py"]
    }
  }
}
"""
import asyncio
import json
import logging
import sys
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError:
    print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Ensure open_brain package is importable
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from open_brain.config import OBSIDIAN_VAULT_PATH
from open_brain.database import KnowledgeBase
from open_brain.embeddings import EmbeddingService

logger = logging.getLogger("open_brain.mcp")

# --- Singletons ---

_kb = None
_embed = None
_kg = None


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def _get_embed() -> EmbeddingService:
    global _embed
    if _embed is None:
        _embed = EmbeddingService(kb=_get_kb())
    return _embed


def _get_kg():
    """Lazy-init knowledge graph — import deferred to avoid circular loads."""
    global _kg
    if _kg is None:
        from open_brain.knowledge_graph import KnowledgeGraph
        _kg = KnowledgeGraph(kb=_get_kb())
    return _kg


# --- MCP Server ---

server = Server("open-brain")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="brain_search",
            description=(
                "Search the agent's persistent memory using semantic similarity. "
                "Use this to find past conversations, decisions, knowledge notes, "
                "or information without relying on context window limits.\n\n"
                "HOW SEARCH WORKS:\n"
                "- Content is stored as ~500-char chunks with 1024-dim embeddings\n"
                "- Results return matching chunks (not full documents), each with "
                "a similarity score, content preview (300 chars), and metadata\n"
                "- Metadata includes file_path (for Obsidian notes), chunk_index, "
                "and total_chunks — use these to understand if you're seeing part "
                "of a larger document\n"
                "- Results with similarity > 50% are returned, ranked highest first\n\n"
                "TIPS:\n"
                "- Use natural language queries — the model understands semantics\n"
                "- If a result shows chunk_index: 2 of total_chunks: 5, there's more "
                "content in that document — search again with different terms to find "
                "other chunks, or use brain_context with the message_id\n"
                "- For Obsidian notes, the file_path in metadata tells you the full "
                "document path (e.g. 'Engineering/my-notes.md')"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you're looking for. Use natural language."
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum results to return"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="brain_recent",
            description=(
                "Fetch recent messages from a chat room. "
                "Use this to get the latest context without semantic search. "
                "Returns full message content (up to 200 chars preview), sender info, "
                "timestamps, and metadata. Messages are ordered chronologically."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "default": "general",
                        "enum": ["general", "builds", "alerts"],
                        "description": "Which room to fetch messages from"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Number of recent messages to fetch"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="brain_upsert",
            description=(
                "Create or update an Obsidian note in Open Brain. "
                "Use this to write daily notes, capture meeting notes, "
                "or save any structured knowledge. The note is stored in "
                "the brain database, embedded for semantic search, and "
                "synced to the Obsidian vault.\n\n"
                "CONTENT LIMITS & CHUNKING:\n"
                "- Content is split into ~500-char chunks for embedding\n"
                "- Each chunk is embedded independently (1024-dim vector)\n"
                "- There is NO hard size limit — large notes just produce more "
                "chunks and take longer to embed\n"
                "- However, very large notes (>5000 chars) are slower to upsert "
                "because each chunk is embedded sequentially\n\n"
                "STRATEGY FOR LARGE CONTENT:\n"
                "- For content that exceeds ~3000 chars, split it across multiple "
                "notes using a pagination convention:\n"
                "  'Topic/my-topic-part-1.md', 'Topic/my-topic-part-2.md', etc.\n"
                "- Each part should be self-contained with its own title and context "
                "so chunks are meaningful when found via search\n"
                "- Add cross-references: 'See also: my-topic-part-2.md' at the end\n"
                "- Prefer multiple focused notes over one giant note — smaller notes "
                "produce higher-quality search results\n\n"
                "UPDATING EXISTING NOTES:\n"
                "- Upsert replaces the entire note content and re-embeds all chunks\n"
                "- IMPORTANT: If updating a note, always read the current content "
                "from the vault first to avoid losing existing information. Do NOT "
                "rely on search results (which are 300-char chunk previews)\n\n"
                "BEST PRACTICES:\n"
                "- Use clear headings and structure — chunks split on line boundaries, "
                "so well-structured markdown produces better chunks\n"
                "- Include relevant keywords naturally — they improve search recall\n"
                "- Use the folder parameter to match the file_path prefix for organization\n"
                "- Tag notes with #hashtags in the content for categorical search"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path within the vault, e.g. 'Daily/2026-03-05.md' "
                            "or 'Meeting Notes/standup.md'. For multi-part notes, "
                            "use 'Topic/name-part-1.md', 'Topic/name-part-2.md', etc."
                        )
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Full markdown content of the note. Aim for ~1000-3000 chars "
                            "per note for optimal search quality. If content exceeds this, "
                            "split across multiple notes with cross-references."
                        )
                    },
                    "folder": {
                        "type": "string",
                        "default": "",
                        "description": "Folder tag for organization, e.g. 'Daily', 'Engineering', 'Meeting Notes'"
                    }
                },
                "required": ["file_path", "content"]
            }
        ),
        types.Tool(
            name="brain_task_create",
            description=(
                "Create a new persistent task in Open Brain. "
                "Tasks survive across sessions and are searchable via brain_search. "
                "Use this for tracking work items, TODOs, follow-ups, and action items "
                "that outlive a single conversation.\n\n"
                "Returns the task with its short ID (e.g. TASK-1) for easy reference."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief task title (max 500 chars)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description, acceptance criteria, context"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Task priority"
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name (e.g. 'buddychat', 'qsic-data', 'chronos')"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    }
                },
                "required": ["summary"]
            }
        ),
        types.Tool(
            name="brain_task_list",
            description=(
                "List tasks from Open Brain. "
                "By default shows active tasks (open, in_progress, blocked). "
                "Use status='all' to see everything including done tasks, "
                "or filter by specific status or project."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "in_progress", "done", "blocked", "all"],
                        "description": "Filter by status. Default: active (non-done) tasks"
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum tasks to return"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="brain_task_update",
            description=(
                "Update an existing task in Open Brain. "
                "Identify the task by short ID (e.g. 'TASK-1') or UUID. "
                "Only provide fields you want to change. "
                "If summary or description changes, the task is re-embedded for search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Task short ID (TASK-1) or UUID"
                    },
                    "summary": {
                        "type": "string",
                        "description": "New task summary"
                    },
                    "description": {
                        "type": "string",
                        "description": "New detailed description"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["open", "in_progress", "done", "blocked"],
                        "description": "New status"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "New priority"
                    },
                    "project": {
                        "type": "string",
                        "description": "New project name"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (replaces existing)"
                    }
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="brain_context",
            description=(
                "Get conversation context around a specific topic or message. "
                "Returns the anchor message plus surrounding messages from the same room. "
                "Useful for understanding the full conversation around a search result. "
                "If you get a brain_search hit and want more context, pass its message_id here."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to find context for."
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Optional: Get context around a specific message ID"
                    },
                    "before": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of messages before to include"
                    },
                    "after": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of messages after to include"
                    }
                },
                "required": ["topic"]
            }
        ),

        # --- Knowledge Graph tools ---

        types.Tool(
            name="brain_fact_store",
            description=(
                "Store a structured fact in the knowledge graph as a temporal triple. "
                "Facts have the form: (subject, predicate, object) with time validity.\n\n"
                "Examples:\n"
                "- (TJ, works_at, QSIC) — entity-to-entity relationship\n"
                "- (auth_service, port, '8080') — entity-to-value attribute\n"
                "- (auth_service, uses, OAuth2) — replacing a previous fact\n\n"
                "ENTITY AUTO-CREATION:\n"
                "- Subject and object entities are automatically created if they don't exist\n"
                "- Entity names are matched case-insensitively\n"
                "- Provide subject_type/object_type hints to categorize new entities\n\n"
                "TEMPORAL INVALIDATION:\n"
                "- By default, storing a fact with the same subject+predicate invalidates "
                "the previous current fact (sets its valid_to to now)\n"
                "- Set invalidate_existing=false for multi-valued predicates like "
                "'knows' or 'depends_on'\n"
                "- Old facts are never deleted — they remain in history with valid_to set"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Subject entity name (e.g. 'TJ', 'auth_service', 'DT-105')"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Relationship type (e.g. 'works_at', 'uses', 'depends_on', 'located_in')"
                    },
                    "object_entity": {
                        "type": "string",
                        "description": "Object entity name, for entity-to-entity facts (e.g. 'QSIC', 'OAuth2')"
                    },
                    "object_value": {
                        "type": "string",
                        "description": "Literal value, for entity-to-value facts (e.g. '8080', 'true', 'Melbourne')"
                    },
                    "subject_type": {
                        "type": "string",
                        "description": "Entity type hint for subject (person, service, project, concept, tool, org)"
                    },
                    "object_type": {
                        "type": "string",
                        "description": "Entity type hint for object entity"
                    },
                    "confidence": {
                        "type": "number",
                        "default": 1.0,
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in this fact (0-1)"
                    },
                    "invalidate_existing": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Auto-invalidate previous facts with same subject+predicate. "
                            "Set false for multi-valued predicates like 'knows' or 'depends_on'."
                        )
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Optional message ID this fact was extracted from"
                    },
                },
                "required": ["subject", "predicate"],
            }
        ),
        types.Tool(
            name="brain_fact_query",
            description=(
                "Query current facts about an entity from the knowledge graph. "
                "Returns facts where the entity appears as either subject or object.\n\n"
                "Supports point-in-time queries: set 'at' to see what was true at a "
                "specific date. Set include_invalid=true to also see superseded facts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to query (case-insensitive)"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Optional: filter to specific predicate"
                    },
                    "at": {
                        "type": "string",
                        "description": "Optional: ISO datetime for point-in-time query (e.g. '2025-06-01T00:00:00')"
                    },
                    "include_invalid": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include superseded/invalidated facts"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["entity"],
            }
        ),
        types.Tool(
            name="brain_fact_history",
            description=(
                "Get the full temporal history of facts about an entity. "
                "Shows all facts including invalidated ones, ordered by valid_from date. "
                "Use this to see how facts have changed over time — e.g., tracking which "
                "technology a service used at different points in time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name (case-insensitive)"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Optional: filter to specific predicate"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": ["entity"],
            }
        ),
        types.Tool(
            name="brain_entity_list",
            description=(
                "List known entities in the knowledge graph. "
                "Use this to discover what entities exist before querying facts. "
                "Filter by type (person, service, project, etc.) or search by name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Filter by entity type (person, service, project, concept, tool, org)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search filter on entity name (case-insensitive substring match)"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": [],
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        if name == "brain_search":
            return await _search_memory(arguments)
        elif name == "brain_recent":
            return await _get_recent(arguments)
        elif name == "brain_upsert":
            return await _upsert_note(arguments)
        elif name == "brain_task_create":
            return await _task_create(arguments)
        elif name == "brain_task_list":
            return await _task_list(arguments)
        elif name == "brain_task_update":
            return await _task_update(arguments)
        elif name == "brain_context":
            return await _get_context(arguments)
        elif name == "brain_fact_store":
            return await _fact_store(arguments)
        elif name == "brain_fact_query":
            return await _fact_query(arguments)
        elif name == "brain_fact_history":
            return await _fact_history(arguments)
        elif name == "brain_entity_list":
            return await _entity_list(arguments)
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e)
        return [types.TextContent(type="text", text=f"Error in {name}: {e}")]


async def _search_memory(args: dict) -> list[types.TextContent]:
    query = args.get("query", "")
    limit = args.get("limit", 5)

    if not query:
        return [types.TextContent(type="text", text="Error: 'query' parameter is required")]

    embed = _get_embed()
    embedding = await embed.generate_embedding(query, is_query=True)

    if not embedding:
        return [types.TextContent(
            type="text",
            text="Could not generate embedding. Is the embedding server running?"
        )]

    kb = _get_kb()
    results = await kb.semantic_search(query, embedding, limit)

    if not results:
        return [types.TextContent(type="text", text=f"No relevant memories found for: {query}")]

    lines = [f"-- Memory Search ({len(results)} results) --"]
    for i, result in enumerate(results, 1):
        lines.append(f"\n[{i}] Similarity: {result['similarity']:.2%}")
        lines.append(f"Content: {result['content'][:300]}")
        if result.get("metadata"):
            lines.append(f"Metadata: {json.dumps(result['metadata'])}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _get_recent(args: dict) -> list[types.TextContent]:
    room = args.get("room", "general")
    limit = args.get("limit", 20)

    kb = _get_kb()
    messages = await kb.get_recent_messages(room, limit)

    if not messages:
        return [types.TextContent(type="text", text=f"No messages found in #{room}")]

    lines = [f"-- Recent Messages in #{room} ({len(messages)} total) --"]
    for msg in messages:
        ts = msg["created_at"][:16].replace("T", " ") if msg.get("created_at") else "unknown"
        lines.append(f"[{ts}] {msg['sender_name']}: {msg['content'][:200]}")
        if msg.get("metadata"):
            lines.append(f"  Metadata: {json.dumps(msg['metadata'])}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _get_context(args: dict) -> list[types.TextContent]:
    topic = args.get("topic", "")
    message_id = args.get("message_id")
    before = args.get("before", 3)
    after = args.get("after", 3)

    if not topic and not message_id:
        return [types.TextContent(type="text", text="Error: provide 'topic' or 'message_id'")]

    kb = _get_kb()
    embed = _get_embed()

    if message_id:
        anchor = await kb.get_message_by_id(message_id)
        if not anchor:
            return [types.TextContent(type="text", text=f"Message {message_id} not found")]
        room = anchor["room"]
    else:
        embedding = await embed.generate_embedding(topic, is_query=True)
        if not embedding:
            return [types.TextContent(type="text", text="Could not generate embedding")]
        results = await kb.semantic_search(topic, embedding, 1)
        if not results:
            return [types.TextContent(type="text", text=f"No messages found for: {topic}")]
        anchor = await kb.get_message_by_id(results[0]["message_id"])
        if not anchor:
            return [types.TextContent(type="text", text="Anchor message not found")]
        room = anchor["room"]

    all_msgs = await kb.get_recent_messages(room, limit=200)

    anchor_idx = None
    anchor_id = message_id or anchor["id"]
    for i, m in enumerate(all_msgs):
        if m["id"] == anchor_id:
            anchor_idx = i
            break

    if anchor_idx is None:
        return [types.TextContent(type="text", text="Could not locate message in timeline")]

    start = max(0, anchor_idx - before)
    end = min(len(all_msgs), anchor_idx + after + 1)
    context_msgs = all_msgs[start:end]

    lines = [f"-- Context for '{topic or message_id}' ({len(context_msgs)} messages) --"]
    for msg in context_msgs:
        marker = " >>> " if msg["id"] == anchor_id else "     "
        ts = msg["created_at"][:16].replace("T", " ") if msg.get("created_at") else ""
        lines.append(f"{marker}[{ts}] {msg['sender_name']}: {msg['content'][:200]}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _upsert_note(args: dict) -> list[types.TextContent]:
    file_path = args.get("file_path", "")
    content = args.get("content", "")
    folder = args.get("folder", "")

    if not file_path or not content:
        return [types.TextContent(
            type="text", text="Error: 'file_path' and 'content' are required"
        )]

    kb = _get_kb()
    embed = _get_embed()

    msg_id = await kb.upsert_note({
        "file_path": file_path,
        "folder": folder,
        "content": content,
        "file_modified": None,
    })

    # Write to Obsidian vault if configured
    vault_written = False
    if OBSIDIAN_VAULT_PATH:
        try:
            vault_file = Path(OBSIDIAN_VAULT_PATH) / file_path
            vault_file.parent.mkdir(parents=True, exist_ok=True)
            vault_file.write_text(content, encoding="utf-8")
            vault_written = True
            logger.info("Wrote note to vault: %s", vault_file)
        except Exception as e:
            logger.error("Failed to write note to vault: %s", e)

    # Re-embed the note
    await kb.delete_embeddings_for_message(msg_id)
    chunks = await embed.chunk_content(content, max_length=500)
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
                    "file_path": file_path,
                    "folder": folder,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            embedded_count += 1

    return [types.TextContent(
        type="text",
        text=(
            f"Note upserted: {file_path}\n"
            f"Chunks: {len(chunks)}, Embedded: {embedded_count}, "
            f"Vault written: {vault_written}"
        )
    )]


async def _task_create(args: dict) -> list[types.TextContent]:
    summary = args.get("summary", "")
    if not summary:
        return [types.TextContent(type="text", text="Error: 'summary' is required")]

    kb = _get_kb()
    embed = _get_embed()

    task_data = {"summary": summary}
    for field in ("description", "priority", "project", "tags"):
        if args.get(field):
            task_data[field] = args[field]

    task = await kb.create_task(task_data)
    embedded = await embed.embed_task(
        task["id"], task["summary"], task.get("description") or ""
    )

    lines = [
        f"Task created: {task['short_id']}",
        f"Summary: {task['summary']}",
    ]
    if task.get("project"):
        lines.append(f"Project: {task['project']}")
    if task.get("priority"):
        lines.append(f"Priority: {task['priority']}")
    if task.get("tags"):
        lines.append(f"Tags: {', '.join(task['tags'])}")
    lines.append(f"Status: {task['status']}")
    lines.append(f"Embedded: {embedded} chunks")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _task_list(args: dict) -> list[types.TextContent]:
    kb = _get_kb()
    status = args.get("status")
    project = args.get("project")
    limit = args.get("limit", 20)

    tasks = await kb.list_tasks(status=status, project=project, limit=limit)

    if not tasks:
        label = status or "active"
        return [types.TextContent(type="text", text=f"No {label} tasks found.")]

    lines = [f"-- Tasks ({len(tasks)}) --", ""]
    for t in tasks:
        priority_tag = f" [{t['priority']}]" if t.get("priority") else ""
        project_tag = f" ({t['project']})" if t.get("project") else ""
        lines.append(f"{t['short_id']}  {t['status']:<12}{priority_tag}{project_tag}  {t['summary']}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _task_update(args: dict) -> list[types.TextContent]:
    identifier = args.get("id", "")
    if not identifier:
        return [types.TextContent(type="text", text="Error: 'id' is required")]

    kb = _get_kb()
    embed = _get_embed()

    updates = {k: v for k, v in args.items() if k != "id" and v is not None}
    if not updates:
        return [types.TextContent(type="text", text="Error: no fields to update")]

    task = await kb.update_task(identifier, updates)
    if not task:
        return [types.TextContent(type="text", text=f"Task {identifier} not found")]

    if "summary" in updates or "description" in updates:
        await embed.embed_task(
            task["id"], task["summary"], task.get("description") or ""
        )

    lines = [
        f"Task updated: {task['short_id']}",
        f"Summary: {task['summary']}",
        f"Status: {task['status']}",
    ]
    if task.get("project"):
        lines.append(f"Project: {task['project']}")
    if task.get("priority"):
        lines.append(f"Priority: {task['priority']}")

    return [types.TextContent(type="text", text="\n".join(lines))]


# --- Knowledge Graph handlers ---


async def _fact_store(args: dict) -> list[types.TextContent]:
    subject = args.get("subject", "")
    predicate = args.get("predicate", "")
    if not subject or not predicate:
        return [types.TextContent(
            type="text", text="Error: 'subject' and 'predicate' are required"
        )]

    object_entity = args.get("object_entity")
    object_value = args.get("object_value")
    if not object_entity and not object_value:
        return [types.TextContent(
            type="text",
            text="Error: provide either 'object_entity' or 'object_value'"
        )]

    kg = _get_kg()
    result = await kg.store_fact(
        subject=subject,
        predicate=predicate,
        object_entity=object_entity,
        object_value=object_value,
        subject_type=args.get("subject_type"),
        object_type=args.get("object_type"),
        confidence=args.get("confidence", 1.0),
        source_id=args.get("source_id"),
        invalidate_existing=args.get("invalidate_existing", True),
    )

    obj_display = result["object"]
    lines = [
        f"Fact stored: ({result['subject']}, {result['predicate']}, {obj_display})",
        f"Valid from: {result['valid_from']}",
        f"Current: {result['is_current']}",
    ]
    if result.get("invalidated_count", 0) > 0:
        lines.append(f"Invalidated {result['invalidated_count']} previous fact(s)")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _fact_query(args: dict) -> list[types.TextContent]:
    entity = args.get("entity", "")
    if not entity:
        return [types.TextContent(type="text", text="Error: 'entity' is required")]

    # Parse optional datetime
    at = None
    at_str = args.get("at")
    if at_str:
        from datetime import datetime, timezone
        try:
            at = datetime.fromisoformat(at_str)
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
        except ValueError:
            return [types.TextContent(
                type="text", text=f"Error: invalid datetime format: {at_str}"
            )]

    kg = _get_kg()
    facts = await kg.query_facts(
        entity=entity,
        predicate=args.get("predicate"),
        at=at,
        include_invalid=args.get("include_invalid", False),
        limit=args.get("limit", 20),
    )

    if not facts:
        label = f" at {at_str}" if at_str else ""
        return [types.TextContent(
            type="text", text=f"No facts found for '{entity}'{label}"
        )]

    lines = [f"-- Facts about '{entity}' ({len(facts)} results) --"]
    for f in facts:
        status = "CURRENT" if f["is_current"] else f"invalidated {f['valid_to']}"
        lines.append(
            f"\n  ({f['subject']}, {f['predicate']}, {f['object']}) "
            f"[{status}]"
        )
        lines.append(f"  Valid from: {f['valid_from']}")
        if f.get("confidence") and f["confidence"] < 1.0:
            lines.append(f"  Confidence: {f['confidence']:.0%}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _fact_history(args: dict) -> list[types.TextContent]:
    entity = args.get("entity", "")
    if not entity:
        return [types.TextContent(type="text", text="Error: 'entity' is required")]

    kg = _get_kg()
    facts = await kg.fact_history(
        entity=entity,
        predicate=args.get("predicate"),
        limit=args.get("limit", 50),
    )

    if not facts:
        return [types.TextContent(
            type="text", text=f"No history found for '{entity}'"
        )]

    lines = [f"-- Fact history for '{entity}' ({len(facts)} entries) --"]
    for f in facts:
        valid_range = f['valid_from']
        if f['valid_to']:
            valid_range += f" -> {f['valid_to']}"
        else:
            valid_range += " -> present"

        marker = "  " if f["is_current"] else "x "
        lines.append(
            f"\n{marker}({f['subject']}, {f['predicate']}, {f['object']})"
        )
        lines.append(f"  Period: {valid_range}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _entity_list(args: dict) -> list[types.TextContent]:
    kg = _get_kg()
    entities = await kg.list_entities(
        entity_type=args.get("type"),
        query=args.get("query"),
        limit=args.get("limit", 50),
    )

    if not entities:
        return [types.TextContent(type="text", text="No entities found")]

    lines = [f"-- Entities ({len(entities)}) --", ""]
    for e in entities:
        type_tag = f" [{e['type']}]" if e.get("type") else ""
        lines.append(f"{e['name']}{type_tag}  ({e['current_facts']} current facts)")

    return [types.TextContent(type="text", text="\n".join(lines))]


# --- Entry ---

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
