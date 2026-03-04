"""
Obsidian sync for Open Brain

Bidirectional sync:
  - Vault -> OpenBrain: Import .md files as embeddings for semantic search
  - OpenBrain -> Vault: Export daily chat summaries to BuddyChat/ folder
  - Chat -> Vault: Sync chat logs per room
"""
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from open_brain.config import (
    OBSIDIAN_VAULT_PATH, OBSIDIAN_SUMMARY_DIR, OBSIDIAN_TEMPLATE,
    SUMMARY_INTERVAL_HOURS,
)
from open_brain.database import KnowledgeBase

logger = logging.getLogger("open_brain.obsidian")

# Track last sync timestamp in a simple file
_SYNC_TIMESTAMP_FILE = Path(__file__).parent.parent / "data" / ".obsidian_last_sync"


def get_last_sync_timestamp() -> float:
    """Get the last vault import timestamp (epoch seconds)."""
    try:
        return float(_SYNC_TIMESTAMP_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def save_last_sync_timestamp(ts: float = None):
    """Save the current time as last sync timestamp."""
    _SYNC_TIMESTAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SYNC_TIMESTAMP_FILE.write_text(str(ts or datetime.now(timezone.utc).timestamp()))


# --- Vault Import (Obsidian -> OpenBrain) ---


class VaultImporter:
    """Imports Obsidian vault .md files into OpenBrain for semantic search."""

    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = Path(vault_path or OBSIDIAN_VAULT_PATH)
        self.kb = KnowledgeBase()
        self._embed_service = None

    def _get_embed_service(self):
        if self._embed_service is None:
            from open_brain.embeddings import EmbeddingService
            self._embed_service = EmbeddingService(kb=self.kb)
        return self._embed_service

    async def import_vault(self, force_full: bool = False) -> dict:
        """
        Import all .md files from the vault into OpenBrain.

        Args:
            force_full: If True, re-import all files regardless of modification time.

        Returns:
            Stats dict with imported/skipped/error counts.
        """
        if not self.vault_path.exists():
            logger.error("Obsidian vault not found: %s", self.vault_path)
            return {"error": f"Vault not found: {self.vault_path}"}

        last_sync = get_last_sync_timestamp() if not force_full else 0.0
        stats = {"imported": 0, "skipped": 0, "errors": 0, "updated": 0}
        embed_svc = self._get_embed_service()

        md_files = list(self.vault_path.rglob("*.md"))
        logger.info("Found %d .md files in vault", len(md_files))

        for md_file in md_files:
            try:
                if md_file.stat().st_size == 0:
                    stats["skipped"] += 1
                    continue

                relative = str(md_file.relative_to(self.vault_path))
                if relative.startswith("BuddyChat/"):
                    stats["skipped"] += 1
                    continue

                file_mtime = md_file.stat().st_mtime
                if not force_full and file_mtime < last_sync:
                    stats["skipped"] += 1
                    continue

                content = md_file.read_text(encoding="utf-8", errors="replace")
                if not content.strip():
                    stats["skipped"] += 1
                    continue

                folder = str(md_file.parent.relative_to(self.vault_path))
                if folder == ".":
                    folder = ""

                existing = await self.kb.find_note_by_path(relative)

                msg_id = await self.kb.upsert_note({
                    "file_path": relative,
                    "folder": folder,
                    "content": content,
                    "file_modified": file_mtime,
                })

                if existing:
                    await self.kb.delete_embeddings_for_message(msg_id)
                    stats["updated"] += 1
                else:
                    stats["imported"] += 1

                chunks = await embed_svc.chunk_content(content, max_length=500)
                for i, chunk in enumerate(chunks):
                    embedding = await embed_svc.generate_embedding(chunk)
                    if embedding:
                        await self.kb.store_embedding(
                            message_id=msg_id,
                            content=chunk,
                            embedding=embedding,
                            metadata={
                                "source": "obsidian",
                                "file_path": relative,
                                "folder": folder,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                            }
                        )

                if (stats["imported"] + stats["updated"]) % 25 == 0:
                    logger.info("Progress: %d imported, %d updated, %d skipped",
                                stats["imported"], stats["updated"], stats["skipped"])

            except Exception as e:
                logger.error("Failed to import %s: %s", md_file.name, e)
                stats["errors"] += 1

        save_last_sync_timestamp()
        logger.info("Vault import complete: %s", stats)
        return stats


# --- Daily Summary Export (OpenBrain -> Obsidian) ---


class SummaryExporter:
    """Exports daily chat summaries to the Obsidian vault."""

    def __init__(self, vault_path: Optional[str] = None):
        summary_dir = vault_path or OBSIDIAN_SUMMARY_DIR
        self.summary_dir = Path(summary_dir)
        self.kb = KnowledgeBase()

    async def write_daily_summary(self) -> Optional[str]:
        """
        Write/update today's summary file with recent chat activity.
        Groups messages by time window and room.
        """
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now(timezone.utc)
        filename = f"{today.strftime('%Y-%m-%d')}.md"
        filepath = self.summary_dir / filename

        rooms = ["general", "builds", "alerts"]
        all_messages = []

        for room in rooms:
            messages = await self.kb.get_recent_messages(room, limit=200)
            day_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
            for msg in messages:
                if msg.get("created_at"):
                    try:
                        msg_dt = datetime.fromisoformat(
                            msg["created_at"].replace('Z', '+00:00')
                        )
                        if msg_dt >= day_start:
                            msg["_room"] = room
                            msg["_dt"] = msg_dt
                            all_messages.append(msg)
                    except (ValueError, TypeError):
                        continue

        if not all_messages:
            logger.info("No messages today — skipping summary")
            return None

        all_messages.sort(key=lambda m: m["_dt"])

        lines = [
            f"# BuddyChat Daily Summary — {today.strftime('%Y-%m-%d')}",
            "",
            f"*Last updated: {today.strftime('%H:%M UTC')}*",
            "",
            "## Activity Log",
            "",
        ]

        window_hours = SUMMARY_INTERVAL_HOURS
        current_window = None

        for msg in all_messages:
            window_start = msg["_dt"].replace(
                hour=(msg["_dt"].hour // window_hours) * window_hours,
                minute=0, second=0, microsecond=0
            )
            window_end = window_start + timedelta(hours=window_hours)
            window_key = f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}"

            if window_key != current_window:
                current_window = window_key
                lines.append(f"### {window_key}")
                lines.append("")

            room_tag = f"**#{msg['_room']}**"
            sender = msg["sender_name"]
            content = msg["content"][:200]
            if len(msg["content"]) > 200:
                content += "..."
            lines.append(f"- {room_tag} **{sender}**: {content}")

        lines.extend(["", "---", f"*Generated by Open Brain at {today.isoformat()}*", ""])

        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Daily summary written to %s (%d messages)", filepath, len(all_messages))
        return str(filepath)


# --- Chat Log Sync ---


class ObsidianSync:
    """Syncs chat history to Obsidian vault as markdown notes."""

    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = Path(vault_path or OBSIDIAN_VAULT_PATH)
        self.template_path = OBSIDIAN_TEMPLATE
        self.kb = KnowledgeBase()

        if not self.vault_path.exists():
            logger.warning("Obsidian vault not found: %s", self.vault_path)

    def generate_note_content(
        self,
        messages: list[dict],
        room: str,
        date: datetime
    ) -> str:
        template = self._load_template()

        now_iso = datetime.now(timezone.utc).isoformat()
        lines = [
            "---",
            f"title: '{room.capitalize()} - {date.strftime('%Y-%m-%d')}'",
            f"tags: [chat/{room}, brain/synced]",
            f"created: {date.isoformat()}",
            f"synced: {now_iso}",
            "---",
            "",
            f"# {room.capitalize()} - {date.strftime('%B %d, %Y')}",
            "",
            f"*Synced from Open Brain at {now_iso}*",
            ""
        ]

        current_sender = None
        for msg in messages:
            if msg["sender_name"] != current_sender:
                lines.append(f"## {msg['sender_name']} ({msg['sender_type']})")
                current_sender = msg["sender_name"]
                lines.append("")

            ts = self._format_timestamp(msg.get("created_at"))
            content = self._format_content(msg["content"])
            lines.append(f"> [{ts}] {content}")
            lines.append("")

        return template + "\n".join(lines)

    def _load_template(self) -> str:
        if self.template_path and self.template_path.exists():
            return self.template_path.read_text()
        return ""

    def _format_timestamp(self, ts: Optional[str]) -> str:
        if not ts:
            return "unknown"
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.strftime('%H:%M')
        except (ValueError, TypeError):
            return str(ts)[:16].replace('T', ' ')

    def _format_content(self, content: str) -> str:
        code_blocks = []

        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        content = re.sub(r'```[\s\S]*?```', save_code_block, content)
        content = content.replace('[', '\\[')
        content = content.replace(']', '\\]')
        content = content.replace('*', '\\*')
        content = content.replace('_', '\\_')

        for i, block in enumerate(code_blocks):
            content = content.replace(f"__CODE_BLOCK_{i}__", block)

        return content

    async def sync_room_to_obsidian(self, room: str = "general", days: int = 7) -> Optional[str]:
        if not self.vault_path.exists():
            logger.error("Obsidian vault not found: %s", self.vault_path)
            return None

        messages = await self.kb.get_recent_messages(room, limit=100)
        if not messages:
            return None

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_messages = [
            m for m in messages
            if m.get("created_at") and self._parse_date(m["created_at"]) >= cutoff
        ]
        if not recent_messages:
            return None

        note_content = self.generate_note_content(
            recent_messages, room, datetime.now(timezone.utc)
        )

        filename = f"{room.capitalize()}-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
        output_dir = self.vault_path / "Chat Logs" / room
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        filepath.write_text(note_content, encoding="utf-8")
        logger.info("Synced %d messages to %s", len(recent_messages), filepath)
        return str(filepath)

    def _parse_date(self, ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except ValueError:
            return datetime.now(timezone.utc)

    async def sync_all_rooms(self, days: int = 7) -> list[str]:
        rooms = ["general", "builds", "alerts"]
        synced = []
        for room in rooms:
            result = await self.sync_room_to_obsidian(room, days)
            if result:
                synced.append(result)
        return synced


# --- Convenience functions ---


async def import_vault(force_full: bool = False) -> dict:
    """Import Obsidian vault into OpenBrain."""
    importer = VaultImporter()
    return await importer.import_vault(force_full)


async def write_daily_summary() -> Optional[str]:
    """Write/update today's daily summary."""
    exporter = SummaryExporter()
    return await exporter.write_daily_summary()


async def sync_all_rooms(days: int = 7) -> list[str]:
    sync = ObsidianSync()
    return await sync.sync_all_rooms(days)


async def sync_room(room: str, days: int = 7) -> Optional[str]:
    sync = ObsidianSync()
    return await sync.sync_room_to_obsidian(room, days)


# --- CLI interface ---

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Open Brain Obsidian Sync")
    sub = parser.add_subparsers(dest="command")

    imp = sub.add_parser("import", help="Import Obsidian vault into OpenBrain")
    imp.add_argument("--full", action="store_true", help="Force full re-import")
    imp.add_argument("--vault", help="Obsidian vault path")

    sub.add_parser("summary", help="Write/update today's daily summary")

    sync_p = sub.add_parser("sync", help="Sync chat logs to Obsidian")
    sync_p.add_argument("--room", "-r", help="Room to sync")
    sync_p.add_argument("--days", "-d", type=int, default=7, help="Days of history")
    sync_p.add_argument("--all", "-a", action="store_true", help="Sync all rooms")
    sync_p.add_argument("--vault", "-v", help="Obsidian vault path")

    args = parser.parse_args()

    async def main():
        if args.command == "import":
            importer = VaultImporter(vault_path=args.vault)
            stats = await importer.import_vault(force_full=args.full)
            print(f"\nImport complete: {stats}")

        elif args.command == "summary":
            result = await write_daily_summary()
            if result:
                print(f"\nSummary written to: {result}")
            else:
                print("\nNo messages to summarize today")

        elif args.command == "sync":
            sync = ObsidianSync(vault_path=args.vault)
            if args.all:
                synced = await sync.sync_all_rooms(days=args.days)
                print(f"\nSynced {len(synced)} rooms")
            elif args.room:
                result = await sync.sync_room_to_obsidian(args.room, args.days)
                if result:
                    print(f"\nSynced to {result}")
                else:
                    print("\nNo messages to sync")
            else:
                parser.print_help()
        else:
            parser.print_help()

    asyncio.run(main())
