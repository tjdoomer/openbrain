"""
py2app build config for Open Brain menu bar app.

Build:
    cd ~/open-brain/launcher
    python setup.py py2app

Output:
    dist/Open Brain.app

Note: The .app only bundles the menu bar UI (rumps, requests, psutil).
      The server (uvicorn, open_brain) runs via the system Python.
"""

from setuptools import setup

APP = ["open_brain_app.py"]
APP_NAME = "Open Brain"

DATA_FILES = [
    ("icons", ["icons/brain.png", "icons/brain@2x.png"]),
]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "icons/OpenBrain.icns",
    "plist": {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "com.openbrain.app",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSUIElement": True,  # Agent app — no Dock icon, menu bar only
    },
    "packages": [
        "rumps",
        "requests",
        "psutil",
    ],
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
