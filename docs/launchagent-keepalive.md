# LaunchAgent with KeepAlive — Auto-Start & Recovery

macOS LaunchAgent that starts the Open Brain API server at login and automatically restarts it if it crashes. Combined with the optional menu bar app, this gives you both reliability and visibility.

## How It Works

```
Boot → Docker Desktop starts → PGVector container resumes (restart: unless-stopped)
     → User login → LaunchAgent fires → uvicorn starts on :8766
     → If uvicorn dies → launchd restarts it after 5s throttle
     → Menu bar app (optional) detects the running server and shows status
```

The LaunchAgent runs uvicorn directly. `KeepAlive: true` tells launchd to restart the process unconditionally if it exits. The menu bar app monitors health and provides start/stop controls via `launchctl`.

## Prerequisites

- Python 3.13+ (system or framework install)
- Dependencies installed: `pip install -r requirements.txt`
- MLX embedding model downloaded to `<repo>/models/qwen3-embedding-0.6b/`
- PGVector running (Docker container with `restart: unless-stopped`)
- Docker Desktop set to launch at login (default behavior)

## Plist Location

```
~/Library/LaunchAgents/com.tjdoomer.openbrain.plist
```

A template is provided at `launcher/com.tjdoomer.openbrain.plist`.

## Plist Contents

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tjdoomer.openbrain</string>

    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python3</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>open_brain.api:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8766</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/path/to/open-brain</string>

    <!-- Start uvicorn when the agent is loaded (i.e. at login) -->
    <key>RunAtLoad</key>
    <true/>

    <!-- Restart unconditionally if the process exits for any reason -->
    <key>KeepAlive</key>
    <true/>

    <!-- Minimum seconds between restart attempts — prevents tight crash loops -->
    <key>ThrottleInterval</key>
    <integer>5</integer>

    <key>StandardOutPath</key>
    <string>/path/to/open-brain/logs/api-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/open-brain/logs/api-stderr.log</string>

    <!-- launchd doesn't read .env files, so all env vars must be declared here -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>BRAIN_HOST</key>
        <string>0.0.0.0</string>
        <key>BRAIN_PORT</key>
        <string>8766</string>
        <key>USE_POSTGRES</key>
        <string>true</string>
        <key>PGVECTOR_HOST</key>
        <string>localhost</string>
        <key>PGVECTOR_PORT</key>
        <string>5432</string>
        <key>PGVECTOR_DB</key>
        <string>braindb</string>
        <key>PGVECTOR_USER</key>
        <string>brainuser</string>
        <key>PGVECTOR_PASSWORD</key>
        <string>YOUR_PASSWORD</string>
        <key>EMBEDDING_MODEL</key>
        <string>qwen3-embedding-0.6b</string>
        <key>EMBEDDING_MODEL_PATH</key>
        <string>models/qwen3-embedding-0.6b</string>
        <key>EMBEDDING_DIM</key>
        <string>1024</string>
        <key>LOG_LEVEL</key>
        <string>INFO</string>
    </dict>
</dict>
</plist>
```

> Replace `/path/to/python3` with the output of `which python3`, `/path/to/open-brain` with the repo location, and `YOUR_PASSWORD` with the PGVector password.

## Setup

```bash
# 1. Create log directory
mkdir -p ~/open-brain/logs

# 2. Copy the template plist
cp ~/open-brain/launcher/com.tjdoomer.openbrain.plist ~/Library/LaunchAgents/

# 3. Edit the plist — set python path, repo path, and PGVector password
nano ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist

# 4. Kill any manually running instance
pkill -f "uvicorn open_brain.api:app"

# 5. Load the agent
launchctl load ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist

# 6. Verify
curl -s http://localhost:8766/health | python3 -m json.tool
```

## Management Commands

```bash
# Stop the service (won't restart until loaded again)
launchctl unload ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist

# Start the service
launchctl load ~/Library/LaunchAgents/com.tjdoomer.openbrain.plist

# Check if running
launchctl list | grep openbrain

# View logs
tail -f ~/open-brain/logs/api-stderr.log
```

## KeepAlive + Menu Bar App

The recommended setup uses both:

| Layer | Role |
|-------|------|
| **LaunchAgent** | Owns the uvicorn process. Auto-starts at login, auto-restarts on crash. The reliable backbone. |
| **Menu bar app** | Optional monitor. Shows health status, provides start/stop (via `launchctl`), log viewing. |

The menu bar app detects the launchd-managed server via health polling and controls it through `launchctl load/unload` rather than managing subprocesses directly.

## Gotchas

- **launchd ignores `.env` files.** Every env var the app reads must be declared in the plist's `EnvironmentVariables` dict. If you add a new env var to `.env`, you must also add it to the plist and reload.
- **Log rotation.** launchd doesn't rotate logs. The stdout/stderr files will grow indefinitely. Consider a periodic cleanup or `newsyslog` config.
- **Boot race condition.** On a fresh boot, Docker Desktop may take 30-60 seconds to start. The uvicorn process will crash on first attempt (PGVector unreachable), but `KeepAlive` + `ThrottleInterval` means launchd retries every 5 seconds until Docker is ready. This is fine — check logs if you want to confirm.
- **Plist changes require reload.** After editing the plist: `launchctl unload` then `launchctl load`.
