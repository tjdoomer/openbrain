"""
Open Brain — macOS Menu Bar App

Monitors the Open Brain uvicorn server (managed by launchd) from the menu bar.
Health status, start/stop via launchctl, log viewing, login item management.

The LaunchAgent (com.tjdoomer.openbrain) owns the uvicorn process with KeepAlive.
This app is an optional overlay for visibility and control — not load-bearing.
"""

import logging
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import requests
import rumps

# Paths
OPEN_BRAIN_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = OPEN_BRAIN_DIR / "logs"
LOG_FILE_STDERR = LOG_DIR / "api-stderr.log"
LOG_FILE_STDOUT = LOG_DIR / "api-stdout.log"
PLIST_NAME = "com.tjdoomer.openbrain"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"
DOCKER_CONTAINER = "open-brain-pgvector"

SERVER_HOST = "localhost"
SERVER_PORT = 8766
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
DOCS_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/docs"
POLL_INTERVAL = 5

# App-level log (not the server log)
APP_LOG = LOG_DIR / "menubar.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(APP_LOG),
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("open-brain-launcher")


class OpenBrainApp(rumps.App):
    def __init__(self):
        super().__init__(
            "Open Brain",
            icon=self._icon_path(),
            template=True,
            quit_button=None,
        )

        self._running = False

        # Build menu
        self.status_item = rumps.MenuItem("Open Brain: Checking...", callback=None)
        self.status_item.set_callback(None)

        self.start_item = rumps.MenuItem("Start Server", callback=self.on_start)
        self.stop_item = rumps.MenuItem("Stop Server", callback=self.on_stop)
        self.restart_item = rumps.MenuItem("Restart Server", callback=self.on_restart)

        self.docs_item = rumps.MenuItem("Open API Docs", callback=self.on_open_docs)
        self.logs_item = rumps.MenuItem("View Logs", callback=self.on_view_logs)

        self.login_item = rumps.MenuItem(
            "Launch at Login", callback=self.on_toggle_login
        )
        self.login_item.state = PLIST_PATH.exists()

        self.quit_item = rumps.MenuItem("Quit", callback=self.on_quit)

        self.menu = [
            self.status_item,
            None,
            self.start_item,
            self.stop_item,
            self.restart_item,
            None,
            self.docs_item,
            self.logs_item,
            None,
            self.login_item,
            self.quit_item,
        ]

        self._update_menu_state()

        # Start polling timer
        self._timer = rumps.Timer(self._poll_health, POLL_INTERVAL)
        self._timer.start()

        # Detect if server is already running
        self._detect_existing_server()

    # --- Icon ---

    @staticmethod
    def _icon_path():
        icons_dir = Path(__file__).resolve().parent / "icons"
        icon = icons_dir / "brain.png"
        return str(icon) if icon.exists() else None

    # --- Menu state ---

    def _update_menu_state(self):
        if self._running:
            self.status_item.title = "Open Brain: Running"
            self.start_item.set_callback(None)
            self.stop_item.set_callback(self.on_stop)
            self.restart_item.set_callback(self.on_restart)
            self.title = "\U0001f7e2"  # green circle
        else:
            self.status_item.title = "Open Brain: Stopped"
            self.start_item.set_callback(self.on_start)
            self.stop_item.set_callback(None)
            self.restart_item.set_callback(None)
            self.title = "\u26aa"  # white circle

    def _detect_existing_server(self):
        """Check if a server is already responding on startup."""
        try:
            resp = requests.get(HEALTH_URL, timeout=2)
            if resp.status_code == 200:
                self._running = True
                log.info("Detected existing Open Brain server")
                self._update_menu_state()
        except requests.ConnectionError:
            pass

    # --- Health polling ---

    def _poll_health(self, _timer):
        try:
            resp = requests.get(HEALTH_URL, timeout=3)
            was_running = self._running
            self._running = resp.status_code == 200
            if self._running != was_running:
                self._update_menu_state()
        except requests.ConnectionError:
            if self._running:
                self._running = False
                self._update_menu_state()
                log.warning("Server health check failed — marking as stopped")

    # --- LaunchAgent management ---

    @staticmethod
    def _is_launchagent_loaded():
        """Check if the LaunchAgent is loaded in launchd."""
        try:
            result = subprocess.run(
                ["launchctl", "list", PLIST_NAME],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def _launchctl_load():
        """Load the LaunchAgent plist."""
        if not PLIST_PATH.exists():
            return False
        result = subprocess.run(
            ["launchctl", "load", str(PLIST_PATH)],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0

    @staticmethod
    def _launchctl_unload():
        """Unload the LaunchAgent plist (stops the server)."""
        if not PLIST_PATH.exists():
            return False
        result = subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0

    # --- Docker check ---

    @staticmethod
    def _is_docker_running():
        """Check if the PGVector Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", DOCKER_CONTAINER],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() == "true"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    # --- Server control (via launchctl) ---

    def on_start(self, _sender):
        if self._running:
            return

        if not PLIST_PATH.exists():
            rumps.notification(
                "Open Brain",
                "LaunchAgent not installed",
                f"Copy the plist to {PLIST_PATH} first. "
                "See docs/launchagent-keepalive.md",
            )
            return

        if not self._is_docker_running():
            rumps.notification(
                "Open Brain",
                "PGVector not detected",
                f"Container '{DOCKER_CONTAINER}' isn't running. "
                "Start it with: docker compose up -d",
            )

        log.info("Loading LaunchAgent...")
        if self._launchctl_load():
            rumps.notification("Open Brain", "Server starting...", "")
        else:
            rumps.notification("Open Brain", "Failed to load LaunchAgent", "")
            log.error("launchctl load failed")

    def on_stop(self, _sender):
        log.info("Unloading LaunchAgent...")
        if self._launchctl_unload():
            self._running = False
            self._update_menu_state()
            rumps.notification("Open Brain", "Server stopped", "")
        else:
            rumps.notification("Open Brain", "Failed to stop server", "")
            log.error("launchctl unload failed")

    def on_restart(self, _sender):
        log.info("Restarting server...")
        self._launchctl_unload()
        if self._launchctl_load():
            rumps.notification("Open Brain", "Server restarting...", "")
        else:
            rumps.notification("Open Brain", "Failed to restart", "")

    # --- Utilities ---

    def on_open_docs(self, _sender):
        if self._running:
            webbrowser.open(DOCS_URL)
        else:
            rumps.notification("Open Brain", "Server not running", "Start the server first")

    def on_view_logs(self, _sender):
        if LOG_FILE_STDERR.exists():
            subprocess.Popen(["open", "-a", "Console", str(LOG_FILE_STDERR)])
        elif LOG_FILE_STDOUT.exists():
            subprocess.Popen(["open", "-a", "Console", str(LOG_FILE_STDOUT)])
        else:
            rumps.notification("Open Brain", "No log file", str(LOG_DIR))

    # --- Login item ---

    def on_toggle_login(self, sender):
        if sender.state:
            self._remove_login_item()
            sender.state = False
            rumps.notification("Open Brain", "Login item removed", "")
        else:
            self._create_login_item()
            sender.state = True
            rumps.notification("Open Brain", "Will launch at login", "")

    def _create_login_item(self):
        """Create a LaunchAgent for the menu bar app itself (not the server).

        The server LaunchAgent (com.tjdoomer.openbrain) is separate and
        should be installed via launcher/com.tjdoomer.openbrain.plist.
        This only auto-starts the menu bar monitor app.
        """
        app_plist_name = "com.tjdoomer.openbrain.menubar"
        app_plist_path = Path.home() / "Library" / "LaunchAgents" / f"{app_plist_name}.plist"

        # Determine the app path
        app_bundle = Path(__file__).resolve().parent / "dist" / "Open Brain.app"
        if app_bundle.exists():
            program_args = f"""    <key>ProgramArguments</key>
    <array>
        <string>open</string>
        <string>{app_bundle}</string>
    </array>"""
        else:
            program_args = f"""    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{Path(__file__).resolve()}</string>
    </array>"""

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{app_plist_name}</string>
{program_args}
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
"""
        app_plist_path.parent.mkdir(parents=True, exist_ok=True)
        app_plist_path.write_text(plist_content)
        log.info(f"Created login item: {app_plist_path}")

    @staticmethod
    def _remove_login_item():
        """Unload and remove the menu bar app LaunchAgent."""
        app_plist_path = Path.home() / "Library" / "LaunchAgents" / "com.tjdoomer.openbrain.menubar.plist"
        if app_plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(app_plist_path)],
                capture_output=True,
            )
            app_plist_path.unlink()
            log.info(f"Removed login item: {app_plist_path}")

    # --- Quit ---

    def on_quit(self, _sender):
        # Menu bar app quits — server stays running under launchd
        rumps.quit_application()


if __name__ == "__main__":
    OpenBrainApp().run()
