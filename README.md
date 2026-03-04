# Open Brain

Standalone knowledge infrastructure for AI agents. PGVector semantic search, local embeddings, and Obsidian vault sync — accessible via REST API or MCP.

Any agent (Claude, LM Studio, custom scripts) can capture memories and search across chat history and notes.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/tjdoomer/openbrain.git ~/open-brain
cd ~/open-brain
cp .env.example .env
pip install -r requirements.txt
```

### 2. Start PGVector

```bash
docker compose up -d
```

This starts PostgreSQL with the pgvector extension on port 5432.

### 3. Load the embedding model

Open LM Studio and load the embedding model:

```bash
lms load qwen3-embedding-0.6b-mxfp8
```

### 4. Start Open Brain

```bash
python -m uvicorn open_brain.api:app --host 0.0.0.0 --port 8766
```

### 5. Test it

```bash
# Capture a memory
curl -X POST http://localhost:8766/api/capture \
  -H "Content-Type: application/json" \
  -d '{"room":"general","sender_id":"test","sender_name":"Test","sender_type":"human","content":"Hello from Open Brain!"}'

# Search
curl "http://localhost:8766/api/search?q=hello&limit=5"
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | PGVector cosine similarity across all captured content |
| **Local Embeddings** | Qwen3-Embedding via LM Studio — no external APIs |
| **Obsidian Sync** | Import vault notes, export daily summaries |
| **REST API** | Capture, search, recent, context, notes upsert |
| **MCP Server** | `brain_search`, `brain_recent`, `brain_context` tools |
| **Auto-chunking** | Long content split into overlapping chunks for better recall |
| **Source Tracking** | Each memory tagged with source, sender, room metadata |

---

## REST API

### Capture a memory

```bash
POST /api/capture
```

```json
{
  "room": "general",
  "sender_id": "agent-1",
  "sender_name": "Claude",
  "sender_type": "claude",
  "content": "We decided to use PostgreSQL for the user store.",
  "metadata": {"topic": "architecture"}
}
```

### Semantic search

```bash
GET /api/search?q=database+decision&limit=5
```

Returns results ranked by cosine similarity:

```json
[
  {
    "content": "We decided to use PostgreSQL for the user store.",
    "similarity": 0.87,
    "metadata": {"source": "chat", "sender_name": "Claude", "room": "general"}
  }
]
```

### Recent messages

```bash
GET /api/recent?limit=20&room=general
```

### Conversation context

```bash
GET /api/context?topic=database&before=3&after=3
```

Returns messages around a topic with surrounding context.

### Upsert notes

```bash
POST /api/notes/upsert
```

```json
{
  "file_path": "dev/architecture-decisions.md",
  "content": "# Architecture Decisions\n\nWe chose PostgreSQL..."
}
```

Used for syncing structured documents (Obsidian notes, dev logs, etc.).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/capture` | Capture a message/memory |
| GET | `/api/search` | Semantic search (`?q=query&limit=N`) |
| GET | `/api/recent` | Recent messages (`?limit=N&room=X`) |
| GET | `/api/context` | Context around a topic (`?topic=X&before=N&after=N`) |
| POST | `/api/notes/upsert` | Upsert a note by file_path |

---

## MCP Server

Open Brain exposes an MCP server for Claude Code, Claude Desktop, or any MCP-compatible client.

### Connect to Claude Code

```bash
claude mcp add open-brain -- python ~/open-brain/open_brain/mcp_server.py
```

### Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "open-brain": {
      "command": "python",
      "args": ["/Users/you/open-brain/open_brain/mcp_server.py"]
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `brain_search` | Semantic search over all memories and notes |
| `brain_recent` | Fetch recent messages from a room |
| `brain_context` | Get conversation context around a topic |

---

## Obsidian Integration

### Import vault notes

Set `OBSIDIAN_VAULT` in `.env` to your vault path, then:

```bash
# Full import (all .md files)
python -m open_brain.obsidian import --full

# Incremental (only modified since last sync)
python -m open_brain.obsidian import
```

### Daily summary export

```bash
python -m open_brain.obsidian summary
```

Writes a summary of chat activity to `BuddyChat/YYYY-MM-DD.md` in your vault.

---

## Integration with BuddyChat

Open Brain was extracted from [BuddyChat](https://github.com/tjdoomer/buddychat) to be a standalone service. BuddyChat integrates with it in two ways:

1. **HTTP** — `_fire_brain_capture()` POSTs every chat message for indexing (fire-and-forget)
2. **MCP Gateway** — Local LLMs in BuddyChat can call `brain_search` etc. via the MCP tool-calling loop

Set `BRAIN_URL=http://localhost:8766` in BuddyChat's `.env`.

---

## Project Structure

```
open-brain/
├── docker-compose.yml         # PGVector container
├── requirements.txt
├── .env.example
└── open_brain/
    ├── __init__.py
    ├── api.py                 # FastAPI REST API
    ├── config.py              # Configuration (env vars)
    ├── database.py            # PGVector ORM + embedding storage
    ├── embeddings.py          # LM Studio embedding client
    ├── mcp_server.py          # MCP stdio server (3 tools)
    └── obsidian.py            # Vault import + summary export
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAIN_HOST` | `0.0.0.0` | Server bind host |
| `BRAIN_PORT` | `8766` | Server port |
| `USE_POSTGRES` | `true` | Enable PGVector (required) |
| `PGVECTOR_HOST` | `localhost` | PostgreSQL host |
| `PGVECTOR_PORT` | `5432` | PostgreSQL port |
| `PGVECTOR_DB` | `braindb` | Database name |
| `PGVECTOR_USER` | `brainuser` | Database user |
| `PGVECTOR_PASSWORD` | _(empty)_ | Database password |
| `EMBEDDING_API_URL` | `http://localhost:1234/v1/embeddings` | Embedding endpoint |
| `EMBEDDING_MODEL` | `qwen3-embedding-0.6b-mxfp8` | Embedding model ID |
| `OBSIDIAN_VAULT` | _(empty)_ | Path to Obsidian vault |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ BuddyChat   │────▶│  Open Brain  │────▶│  PGVector   │
│ Claude Code  │     │  (FastAPI)   │     │ (Postgres)  │
│ Any Agent    │◀────│              │     └─────────────┘
└─────────────┘     │  Embeddings  │
       REST/MCP     │  + Search    │     ┌─────────────┐
                    │              │────▶│  LM Studio  │
                    │  Obsidian    │     │ (embeddings)│
                    │  Sync        │     └─────────────┘
                    └──────────────┘
                           │
                    ┌──────▼──────┐
                    │  Obsidian   │
                    │  Vault      │
                    └─────────────┘
```

---

## License

MIT
