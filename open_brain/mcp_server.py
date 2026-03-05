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


# --- MCP Server ---

server = Server("open-brain")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="brain_search",
            description=(
                "Search the agent's memory using semantic similarity. "
                "Use this to find past conversations, decisions, or information "
                "without relying on context window limits."
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
                "Use this to get the latest context without semantic search."
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
                "synced to the Obsidian vault."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path within the vault, e.g. 'Daily/2026-03-05.md' "
                            "or 'Meeting Notes/standup.md'"
                        )
                    },
                    "content": {
                        "type": "string",
                        "description": "Full markdown content of the note"
                    },
                    "folder": {
                        "type": "string",
                        "default": "",
                        "description": "Folder tag for organization, e.g. 'Daily'"
                    }
                },
                "required": ["file_path", "content"]
            }
        ),
        types.Tool(
            name="brain_context",
            description=(
                "Get conversation context around a specific topic or message. "
                "Returns the original message plus surrounding context."
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
        elif name == "brain_context":
            return await _get_context(arguments)
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
    embedding = await embed.generate_embedding(query)

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
        embedding = await embed.generate_embedding(topic)
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
