#!/usr/bin/env python3
"""
Re-embed all existing content with the current embedding model.

Usage:
    python scripts/reembed.py

This iterates all rows in the embeddings table and regenerates
each embedding vector using the configured MLX model.
Run this after switching embedding models to ensure all vectors
are in the same vector space.
"""
import asyncio
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_brain.config import EMBEDDING_MODEL
from open_brain.embeddings import EmbeddingService
from open_brain.database import KnowledgeBase, USE_POSTGRES

BATCH_SIZE = 50


async def main():
    print(f"Re-embedding all content with model: {EMBEDDING_MODEL}")

    embed = EmbeddingService()
    # Warm up the model
    await embed.generate_embedding("warmup")
    print("Model loaded")

    kb = KnowledgeBase()
    session = kb.get_session()

    try:
        from sqlalchemy import text

        total = session.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()
        print(f"Total embeddings to re-generate: {total}")

        if total == 0:
            print("Nothing to re-embed.")
            return

        offset = 0
        updated = 0
        failed = 0

        while offset < total:
            rows = session.execute(
                text(
                    "SELECT id, content FROM embeddings "
                    "ORDER BY id LIMIT :limit OFFSET :offset"
                ),
                {"limit": BATCH_SIZE, "offset": offset},
            ).fetchall()

            if not rows:
                break

            for row in rows:
                emb_id = row[0]
                content = row[1]

                new_vec = await embed.generate_embedding(content[:8000])
                if new_vec is None:
                    failed += 1
                    continue

                if USE_POSTGRES:
                    session.execute(
                        text(
                            "UPDATE embeddings SET embedding = CAST(:vec AS vector) "
                            "WHERE id = :id"
                        ),
                        {"vec": str(new_vec), "id": emb_id},
                    )
                else:
                    import json
                    session.execute(
                        text("UPDATE embeddings SET embedding = :vec WHERE id = :id"),
                        {"vec": json.dumps(new_vec), "id": emb_id},
                    )
                updated += 1

            session.commit()
            offset += BATCH_SIZE
            print(
                f"  Progress: {min(offset, total)}/{total} "
                f"({updated} updated, {failed} failed)"
            )

        print(f"\nDone. Updated: {updated}, Failed: {failed}")

    except Exception as e:
        session.rollback()
        print(f"Error: {e}", file=sys.stderr)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
