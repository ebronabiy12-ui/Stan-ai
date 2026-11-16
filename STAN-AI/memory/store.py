from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MEMORY_CATEGORIES: tuple[str, ...] = (
    "identity_personality",
    "people_relationships",
    "user_facts",
    "conversation_history",
    "project_knowledge",
    "learned_preferences",
    "decisions_experiences",
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset({
    "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "what", "who", "whom", "whose", "when", "where", "why", "how", "which",
    "a", "an", "and", "or", "but", "if", "so", "as", "at", "by", "for",
    "from", "in", "into", "of", "on", "to", "with", "about", "inside",
    "it", "its", "that", "this", "these", "those", "my", "your", "our",
    "me", "you", "we", "they", "them", "do", "does", "did", "can", "could",
    "would", "should", "have", "has", "had", "just", "there", "here",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize_terms(text: str) -> set[str]:
    terms = {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1}
    filtered = terms - _STOPWORDS
    return filtered or terms


@dataclass
class MemoryRecord:
    id: str
    category: str
    subject: str
    content: str
    source: str
    confidence: float
    pinned: bool
    created_at: str
    updated_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """
    Durable SQLite-backed persistent memory architecture separated into 7 explicit stores:
    - identity_personality
    - people_relationships
    - user_facts
    - conversation_history
    - project_knowledge
    - learned_preferences
    - decisions_experiences
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._seed_foundational_records()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject)"
            )
            conn.commit()

    def _seed_foundational_records(self) -> None:
        """
        Seed permanent foundational identity, Stan/Kalda relationship, and project architecture
        only if not yet present. Never overwrites user edits after initial seeding.
        """
        defaults = [
            {
                "id": "core_identity_stan_ai",
                "category": "identity_personality",
                "subject": "STAN AI core identity",
                "content": (
                    "My name is STAN AI. I am an independent AI system built on a custom "
                    "51,658,240-parameter decoder-only Transformer with persistent memory and "
                    "structured reasoning. I am not conscious and do not genuinely experience emotions."
                ),
                "source": "system_foundation",
                "confidence": 1.0,
                "pinned": True,
                "metadata": {"immutable_anchor": True},
            },
            {
                "id": "rel_owner_stan",
                "category": "people_relationships",
                "subject": "Stan (owner and creator)",
                "content": (
                    "Stan is the creator and owner of STAN AI. Stan is Kalda's son. "
                    "When Stan asks 'Who am I?', the established answer is: \"You are Stan, Kalda's son.\""
                ),
                "source": "owner_foundation",
                "confidence": 1.0,
                "pinned": True,
                "metadata": {"person": "Stan", "relation": "Kalda's son"},
            },
            {
                "id": "proj_kalda_stan_ai",
                "category": "project_knowledge",
                "subject": "Kalda project and STAN AI architecture",
                "content": (
                    "STAN AI belongs to Stan and is part of the Kalda project. "
                    "Its neural core is a 15-layer, 512-dim, 8-head, 2048-FFN PyTorch decoder-only "
                    "Transformer (51,658,240 parameters) with an 8,192-vocab SentencePiece BPE tokenizer, "
                    "FastAPI inference backend, and React web interface."
                ),
                "source": "project_foundation",
                "confidence": 1.0,
                "pinned": True,
                "metadata": {"project": "Kalda"},
            },
        ]

        now = _now_iso()
        with self._connect() as conn:
            for rec in defaults:
                existing = conn.execute(
                    "SELECT id FROM memories WHERE id = ?", (rec["id"],)
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO memories (
                            id, category, subject, content, source, confidence,
                            pinned, created_at, updated_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec["id"],
                            rec["category"],
                            rec["subject"],
                            rec["content"],
                            rec["source"],
                            float(rec["confidence"]),
                            1 if rec["pinned"] else 0,
                            now,
                            now,
                            json.dumps(rec["metadata"]),
                        ),
                    )
            conn.commit()

    def _validate_category(self, category: str) -> None:
        if category not in MEMORY_CATEGORIES:
            raise ValueError(
                f"Invalid memory category '{category}'. Allowed: {MEMORY_CATEGORIES}"
            )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            category=row["category"],
            subject=row["subject"],
            content=row["content"],
            source=row["source"],
            confidence=float(row["confidence"]),
            pinned=bool(row["pinned"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def add_memory(
        self,
        category: str,
        subject: str,
        content: str,
        source: str = "user_input",
        confidence: float = 0.9,
        pinned: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> MemoryRecord:
        self._validate_category(category)
        subj = subject.strip()
        body = content.strip()
        if not subj or not body:
            raise ValueError("Memory subject and content must be non-empty.")

        rec_id = record_id or f"mem_{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        conf = min(1.0, max(0.0, float(confidence)))
        meta_json = json.dumps(metadata or {})

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, category, subject, content, source, confidence,
                    pinned, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rec_id, category, subj, body, source, conf, 1 if pinned else 0, now, now, meta_json),
            )
            conn.commit()
        rec = self.get_memory(rec_id)
        assert rec is not None
        return rec

    def get_memory(self, record_id: str) -> Optional[MemoryRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (record_id,)).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_memories(
        self,
        category: Optional[str] = None,
        limit: int = 200,
    ) -> list[MemoryRecord]:
        if category is not None:
            self._validate_category(category)
            sql = (
                "SELECT * FROM memories WHERE category = ? "
                "ORDER BY pinned DESC, updated_at DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (category, limit)
        else:
            sql = (
                "SELECT * FROM memories "
                "ORDER BY pinned DESC, updated_at DESC LIMIT ?"
            )
            params = (limit,)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_memory(
        self,
        record_id: str,
        subject: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        confidence: Optional[float] = None,
        pinned: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[MemoryRecord]:
        existing = self.get_memory(record_id)
        if existing is None:
            return None

        new_cat = category if category is not None else existing.category
        self._validate_category(new_cat)
        new_subj = subject.strip() if subject is not None else existing.subject
        new_body = content.strip() if content is not None else existing.content
        if not new_subj or not new_body:
            raise ValueError("Memory subject and content must remain non-empty.")

        new_conf = (
            min(1.0, max(0.0, float(confidence)))
            if confidence is not None
            else existing.confidence
        )
        new_pinned = bool(pinned) if pinned is not None else existing.pinned
        new_meta = metadata if metadata is not None else existing.metadata
        now = _now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET category = ?, subject = ?, content = ?, confidence = ?,
                    pinned = ?, updated_at = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    new_cat,
                    new_subj,
                    new_body,
                    new_conf,
                    1 if new_pinned else 0,
                    now,
                    json.dumps(new_meta),
                    record_id,
                ),
            )
            conn.commit()
        return self.get_memory(record_id)

    def delete_memory(self, record_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
            conn.commit()
            return cur.rowcount > 0

    def clear_category(self, category: str, keep_pinned: bool = True) -> int:
        self._validate_category(category)
        with self._connect() as conn:
            if keep_pinned:
                cur = conn.execute(
                    "DELETE FROM memories WHERE category = ? AND pinned = 0",
                    (category,),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM memories WHERE category = ?",
                    (category,),
                )
            conn.commit()
            return cur.rowcount

    def search_memories(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> list[MemoryRecord]:
        if categories:
            for c in categories:
                self._validate_category(c)
            all_recs: list[MemoryRecord] = []
            for c in categories:
                all_recs.extend(self.list_memories(category=c, limit=500))
        else:
            all_recs = self.list_memories(limit=500)

        q_terms = _tokenize_terms(query)
        q_lower = query.lower().strip()

        scored: list[tuple[float, MemoryRecord]] = []
        for rec in all_recs:
            text_blob = f"{rec.subject} {rec.content}".lower()
            rec_terms = _tokenize_terms(text_blob)
            overlap = len(q_terms & rec_terms)
            phrase_bonus = 2.5 if (q_lower and q_lower in text_blob) else 0.0
            pinned_bonus = 0.35 if rec.pinned else 0.0
            score = overlap + phrase_bonus + (rec.confidence * 0.2) + pinned_bonus
            if overlap > 0 or phrase_bonus > 0:
                scored.append((score, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def category_counts(self) -> dict[str, int]:
        counts = {cat: 0 for cat in MEMORY_CATEGORIES}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT category, COUNT(*) AS cnt FROM memories GROUP BY category"
            ).fetchall()
        for row in rows:
            if row["category"] in counts:
                counts[row["category"]] = int(row["cnt"])
        return counts
