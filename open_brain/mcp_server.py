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
