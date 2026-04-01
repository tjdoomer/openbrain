# Open Brain — Setup Guide

How to get Open Brain running as a persistent background service on macOS.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| macOS | 13+ (Ventura) | Apple Silicon required for MLX embeddings |
| Python | 3.13+ | System or framework install (`which python3` to verify) |
| Docker Desktop | Latest | Must be set to start at login |
| ~2GB disk | — | Model weights (~1.2GB) + Docker image + DB |

---

## 1. Clone & Configure

```bash
git clone https://github.com/tjdoomer/openbrain.git ~/open-brain
cd ~/open-brain
cp .env.example .env
```

Edit `.env`:

```env
USE_POSTGRES=true
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DB=braindb
PGVECTOR_USER=brainuser
PGVECTOR_PASSWORD=<choose-a-password>

# Optional — set to your Obsidian vault path for note sync
# OBSIDIAN_VAULT=/path/to/your/vault
```

> `.env` is used when running uvicorn manually. The LaunchAgent has its own env vars in the plist (launchd ignores `.env`).

---

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Download the Embedding Model

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir models/qwen3-embedding-0.6b
```

This is an MLX-quantized model (MXFP8) that runs natively on Apple Silicon. It produces 1024-dimensional vectors and loads lazily on first request.

---

## 4. Start PGVector (Docker)

```bash
docker compose up -d
```

Starts PostgreSQL + pgvector on port 5432 with a persistent named volume (`pgdata`). The container uses `restart: unless-stopped` — it survives Docker Desktop restarts automatically.

**Verify:**

```bash
docker ps --filter name=open-brain-pgvector
```

---

## 5. Start the API Server

### LaunchAgent with KeepAlive (recommended)

This is the standard approach. launchd owns the uvicorn process, starts it at login, and auto-restarts it if it crashes.

```bash
# Create log directory
mkdir -p ~/open-brain/logs

# Copy the template plist
cp ~/open-brain/launcher/com.tjdoomer.openbrain.plist ~/Library/LaunchAgents/

# Edit — set your python path, repo path, and PGVector password
nano ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist
```

Replace these placeholders in the plist:
- `/path/to/python3` → output of `which python3`
- `/path/to/open-brain` → your repo location (e.g. `/Users/you/open-brain`)
- `YOUR_PASSWORD` → your PGVector password from `.env`

Then load it:

```bash
# Kill any manually running instance first
pkill -f "uvicorn open_brain.api:app" 2>/dev/null

# Load the LaunchAgent
launchctl load ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist

# Verify
curl -s http://localhost:8766/health | python3 -m json.tool
```

See [docs/launchagent-keepalive.md](docs/launchagent-keepalive.md) for full details and gotchas.

### Manual (for development)

```bash
python -m uvicorn open_brain.api:app --host 0.0.0.0 --port 8766
```

---

## 6. Menu Bar App (optional)

The menu bar app provides a visual monitor for the launchd-managed server: health status indicator, start/stop controls (via `launchctl`), and log access.

```bash
cd ~/open-brain/launcher
pip install -r requirements.txt
python setup.py py2app
cp -r dist/Open\ Brain.app /Applications/
```

The app is not load-bearing — quitting it doesn't stop the server. The LaunchAgent handles reliability.

---

## 7. Register the MCP Server

### Claude Code

```bash
claude mcp add open-brain -- python3 ~/open-brain/open_brain/mcp_server.py
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "open-brain": {
      "command": "python3",
      "args": ["/path/to/open-brain/open_brain/mcp_server.py"]
    }
  }
}
```

Tools exposed: `brain_search`, `brain_recent`, `brain_upsert`, `brain_context`, `brain_task_create`, `brain_task_list`, `brain_task_update`.

---

## 8. Verify

```bash
# Health check
curl -s http://localhost:8766/health | python3 -m json.tool

# Capture a test memory
curl -X POST http://localhost:8766/api/capture \
  -H "Content-Type: application/json" \
  -d '{"room":"general","sender_id":"test","sender_name":"Test","sender_type":"human","content":"Setup verification"}'

# Search for it
curl "http://localhost:8766/api/search?q=setup+verification&limit=5"
```

---

## Architecture

```
Boot sequence:
  1. Docker Desktop starts (macOS login item)
  2. PGVector container auto-starts (restart: unless-stopped)
  3. LaunchAgent starts uvicorn (RunAtLoad + KeepAlive)
  4. uvicorn connects to PGVector, loads MLX model on first request
  5. MCP server spawned on-demand by Claude Code/Desktop
  6. Menu bar app (optional) monitors health via polling

┌────────────────────────────────────────────────────────┐
│ macOS                                                  │
│                                                        │
│  Docker Desktop                                        │
│  └─ open-brain-pgvector (port 5432)                    │
│     └─ PostgreSQL 16 + pgvector extension              │
│     └─ Volume: pgdata (persistent)                     │
│                                                        │
│  LaunchAgent (com.tjdoomer.openbrain)                  │
│  └─ uvicorn open_brain.api:app (port 8766)             │
│     ├─ KeepAlive: true (auto-restart on crash)         │
│     ├─ FastAPI REST API                                │
│     ├─ MLX embeddings (Qwen3-0.6B, Apple Silicon)      │
│     └─ Obsidian vault sync (optional)                  │
│                                                        │
│  Menu bar app (optional monitor)                       │
│  └─ Health polling, launchctl start/stop, log viewer   │
│                                                        │
│  Claude Code / Claude Desktop                          │
│  └─ MCP server (stdio, spawned on demand)              │
└────────────────────────────────────────────────────────┘
```

---

## Management

```bash
# Stop server
launchctl unload ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist

# Start server
launchctl load ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist

# Check status
launchctl list | grep openbrain

# View logs
tail -f ~/open-brain/logs/api-stderr.log

# Restart PGVector
docker compose -f ~/open-brain/docker-compose.yml restart
```

---

## Troubleshooting

**Server won't start:**
```bash
lsof -i :8766                    # check if port is in use
docker ps --filter name=pgvector  # check PGVector
tail -20 ~/open-brain/logs/api-stderr.log
```

**LaunchAgent not running:**
```bash
launchctl list | grep openbrain
# Exit code 0 = running, non-zero = check logs
launchctl unload ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist
launchctl load ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist
```

**Embeddings not loading:**
```bash
ls ~/open-brain/models/qwen3-embedding-0.6b/
# Should contain config.json, model.safetensors, tokenizer.json, etc.
```

**Env var not picked up by launchd:**
launchd ignores `.env`. Add the var to the plist's `EnvironmentVariables` dict and reload.
