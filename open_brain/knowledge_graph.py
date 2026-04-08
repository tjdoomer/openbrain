"""
Open Brain — Temporal Knowledge Graph

Stores entity-relationship triples with time validity windows. When facts
change, old ones get invalidated (valid_to set to now) rather than deleted,
preserving a full history of what was true and when.

Inspired by Zep's Graphiti, implemented in PostgreSQL (no Neo4j).

Usage:
    kg = KnowledgeGraph(kb=knowledge_base_instance)
    await kg.store_fact("auth_service", "uses", object_entity="JWT",
                        subject_type="service", object_type="tool")
    # Later, when migrating:
    await kg.store_fact("auth_service", "uses", object_entity="OAuth2")
    # The JWT fact is auto-invalidated, OAuth2 is current.
    # Both remain in history.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from open_brain.database import KnowledgeBase, Entity, Fact

logger = logging.getLogger("open_brain.kg")


class KnowledgeGraph:
    """Temporal knowledge graph backed by PostgreSQL.

    Manages entities (nodes) and facts (temporal edges/attributes).
    All writes go through a single session per operation to keep
    entity resolution and fact invalidation atomic.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    # --- Async wrappers ---

    async def store_fact(
        self,
        subject: str,
        predicate: str,
        object_entity: Optional[str] = None,
        object_value: Optional[str] = None,
        subject_type: Optional[str] = None,
        object_type: Optional[str] = None,
        valid_from: Optional[datetime] = None,
        confidence: float = 1.0,
        source_id: Optional[str] = None,
        invalidate_existing: bool = True,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Store a fact triple with temporal validity.

        Either object_entity or object_value must be provided (not both).
        Entities are auto-created if they don't exist.
        If invalidate_existing=True (default), any current fact with the
        same subject+predicate gets its valid_to set to now.
        """
        return await asyncio.to_thread(
            self._store_fact_sync,
            subject, predicate, object_entity, object_value,
            subject_type, object_type, valid_from, confidence,
            source_id, invalidate_existing, metadata,
        )

    async def query_facts(
        self,
        entity: str,
        predicate: Optional[str] = None,
        at: Optional[datetime] = None,
        include_invalid: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """Query facts about an entity (as subject or object).

        By default returns only current facts (valid_to IS NULL).
        Set at=datetime to query what was true at a specific point in time.
        Set include_invalid=True to see all facts regardless of validity.
        """
        return await asyncio.to_thread(
            self._query_facts_sync, entity, predicate, at, include_invalid, limit
        )

    async def fact_history(
        self,
        entity: str,
        predicate: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Full timeline of facts for an entity, including invalidated ones.

        Ordered by valid_from descending (newest first).
        """
        return await asyncio.to_thread(
            self._fact_history_sync, entity, predicate, limit
        )

    async def list_entities(
        self,
        entity_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List known entities, optionally filtered by type or name search."""
        return await asyncio.to_thread(
            self._list_entities_sync, entity_type, query, limit
        )

    # --- Sync implementations ---

    def _store_fact_sync(
        self,
        subject: str,
        predicate: str,
        object_entity: Optional[str],
        object_value: Optional[str],
        subject_type: Optional[str],
        object_type: Optional[str],
        valid_from: Optional[datetime],
        confidence: float,
        source_id: Optional[str],
        invalidate_existing: bool,
        metadata: Optional[dict],
    ) -> dict:
        has_entity = object_entity is not None and object_entity != ""
        has_value = object_value is not None and object_value != ""

        if has_entity == has_value:
            raise ValueError(
                "Exactly one of object_entity or object_value must be provided"
            )

        now = datetime.now(timezone.utc)
        if valid_from is None:
            valid_from = now

        predicate_lower = predicate.strip().lower()

        session = self.kb.get_session()
        try:
            # Get-or-create subject entity
            subject_ent = self._get_or_create_entity_sync(
                session, subject, subject_type
            )

            # Get-or-create object entity if it's an entity-to-entity fact
            object_ent = None
            if has_entity:
                object_ent = self._get_or_create_entity_sync(
                    session, object_entity, object_type
                )

            # Invalidate existing current facts with same subject+predicate
            invalidated = []
            if invalidate_existing:
                existing = (
                    session.query(Fact)
                    .filter_by(subject_id=subject_ent.id, predicate=predicate_lower)
                    .filter(Fact.valid_to.is_(None))
                    .all()
                )
                for old_fact in existing:
                    old_fact.valid_to = now
                    invalidated.append(old_fact.id)

            # Insert new fact
            fact_id = str(uuid4())
            fact = Fact(
                id=fact_id,
                subject_id=subject_ent.id,
                predicate=predicate_lower,
                object_entity_id=object_ent.id if object_ent else None,
                object_value=object_value if has_value else None,
                confidence=confidence,
                source_id=source_id,
                valid_from=valid_from,
                valid_to=None,
                meta=metadata,
                created_at=now,
            )
            session.add(fact)
            session.commit()

            result = self._fact_to_dict(fact, subject_ent, object_ent)
            result["invalidated_count"] = len(invalidated)
            return result

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _query_facts_sync(
        self,
        entity: str,
        predicate: Optional[str],
        at: Optional[datetime],
        include_invalid: bool,
        limit: int,
    ) -> list[dict]:
        session = self.kb.get_session()
        try:
            # Resolve entity by name (case-insensitive)
            ent = session.query(Entity).filter_by(name=entity.strip().lower()).first()
            if not ent:
                return []

            # Query facts where entity is subject OR object
            query = session.query(Fact).filter(
                (Fact.subject_id == ent.id) | (Fact.object_entity_id == ent.id)
            )

            if predicate:
                query = query.filter_by(predicate=predicate.strip().lower())

            if not include_invalid:
                if at:
                    # Point-in-time: valid_from <= at AND (valid_to IS NULL OR valid_to > at)
                    query = query.filter(
                        Fact.valid_from <= at,
                        (Fact.valid_to.is_(None)) | (Fact.valid_to > at),
                    )
                else:
                    # Current facts only
                    query = query.filter(Fact.valid_to.is_(None))

            facts = query.order_by(Fact.valid_from.desc()).limit(limit).all()
            return self._resolve_facts(session, facts)

        finally:
            session.close()

    def _fact_history_sync(
        self,
        entity: str,
        predicate: Optional[str],
        limit: int,
    ) -> list[dict]:
        session = self.kb.get_session()
        try:
            ent = session.query(Entity).filter_by(name=entity.strip().lower()).first()
            if not ent:
                return []

            query = session.query(Fact).filter(
                (Fact.subject_id == ent.id) | (Fact.object_entity_id == ent.id)
            )

            if predicate:
                query = query.filter_by(predicate=predicate.strip().lower())

            # All facts, ordered newest first
            facts = query.order_by(Fact.valid_from.desc()).limit(limit).all()
            return self._resolve_facts(session, facts)

        finally:
            session.close()

    def _list_entities_sync(
        self,
        entity_type: Optional[str],
        query_str: Optional[str],
        limit: int,
    ) -> list[dict]:
        session = self.kb.get_session()
        try:
            query = session.query(Entity)

            if entity_type:
                query = query.filter_by(entity_type=entity_type.strip().lower())

            if query_str:
                # Case-insensitive substring match on name or display_name
                pattern = f"%{query_str.strip().lower()}%"
                query = query.filter(Entity.name.ilike(pattern))

            entities = query.order_by(Entity.created_at.desc()).limit(limit).all()

            results = []
            for ent in entities:
                # Count current facts where this entity is the subject
                fact_count = (
                    session.query(Fact)
                    .filter_by(subject_id=ent.id)
                    .filter(Fact.valid_to.is_(None))
                    .count()
                )
                results.append({
                    "id": ent.id,
                    "name": ent.display_name,
                    "type": ent.entity_type,
                    "metadata": ent.meta,
                    "current_facts": fact_count,
                    "created_at": ent.created_at.isoformat() if ent.created_at else None,
                })

            return results

        finally:
            session.close()

    # --- Helpers ---

    def _get_or_create_entity_sync(
        self,
        session,
        name: str,
        entity_type: Optional[str] = None,
    ) -> Entity:
        """Look up entity by lowercased name. Create if missing.

        If the entity exists but has no type and a type is now provided,
        backfill the type — this handles the common case where an entity
        is first referenced as an object (no type hint) and later as a
        subject (with type hint).
        """
        name_lower = name.strip().lower()
        display_name = name.strip()
        type_lower = entity_type.strip().lower() if entity_type else None

        ent = session.query(Entity).filter_by(name=name_lower).first()
        if ent:
            # Backfill type if it was NULL and caller now provides one
            if type_lower and not ent.entity_type:
                ent.entity_type = type_lower
            return ent

        ent = Entity(
            id=str(uuid4()),
            name=name_lower,
            display_name=display_name,
            entity_type=type_lower,
            meta=None,
            created_at=datetime.now(timezone.utc),
        )
        session.add(ent)
        session.flush()  # Assign ID without committing (caller commits)
        return ent

    def _fact_to_dict(
        self,
        fact: Fact,
        subject: Entity,
        object_entity: Optional[Entity] = None,
    ) -> dict:
        """Serialize a fact with resolved entity names."""
        result = {
            "id": fact.id,
            "subject": subject.display_name,
            "subject_type": subject.entity_type,
            "predicate": fact.predicate,
            "confidence": fact.confidence,
            "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
            "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
            "is_current": fact.valid_to is None,
            "source_id": fact.source_id,
            "metadata": fact.meta,
        }

        if object_entity:
            result["object"] = object_entity.display_name
            result["object_type"] = object_entity.entity_type
        else:
            result["object"] = fact.object_value
            result["object_type"] = None

        return result

    def _resolve_facts(self, session, facts: list[Fact]) -> list[dict]:
        """Resolve entity references for a list of facts.

        Batch-loads all referenced entities to avoid N+1 queries.
        """
        if not facts:
            return []

        # Collect all entity IDs referenced by these facts
        entity_ids = set()
        for f in facts:
            entity_ids.add(f.subject_id)
            if f.object_entity_id:
                entity_ids.add(f.object_entity_id)

        # Batch load entities
        entities = session.query(Entity).filter(Entity.id.in_(entity_ids)).all()
        entity_map = {e.id: e for e in entities}

        results = []
        for f in facts:
            subject = entity_map.get(f.subject_id)
            obj_ent = entity_map.get(f.object_entity_id) if f.object_entity_id else None
            if subject:
                results.append(self._fact_to_dict(f, subject, obj_ent))

        return results
