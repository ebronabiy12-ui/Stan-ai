from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import MEMORY_CATEGORIES, MemoryStore


def test_memory_categories_and_foundational_seeding(tmp_path: Path) -> None:
    db_file = tmp_path / "stan_memory.db"
    store = MemoryStore(db_file)

    assert len(MEMORY_CATEGORIES) == 7
    counts = store.category_counts()
    assert set(counts.keys()) == set(MEMORY_CATEGORIES)
    assert counts["identity_personality"] >= 1
    assert counts["people_relationships"] >= 1
    assert counts["project_knowledge"] >= 1

    stan_rec = store.get_memory("rel_owner_stan")
    assert stan_rec is not None
    assert "Kalda's son" in stan_rec.content


def test_memory_crud_search_and_restart_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "stan_memory.db"
    store1 = MemoryStore(db_file)

    created = store1.add_memory(
        category="user_facts",
        subject="Training hardware",
        content="Stan trains checkpoints on a Kaggle T4 GPU.",
        source="user_input",
        confidence=0.95,
    )
    assert created.id.startswith("mem_")

    # Simulate application restart by opening a fresh MemoryStore on the same SQLite DB
    store2 = MemoryStore(db_file)
    reloaded = store2.get_memory(created.id)
    assert reloaded is not None
    assert reloaded.content == "Stan trains checkpoints on a Kaggle T4 GPU."

    # Update / correction
    updated = store2.update_memory(created.id, content="Stan upgraded to a Kaggle P100 GPU.")
    assert updated is not None
    assert "P100" in updated.content

    # Retrieval search
    hits = store2.search_memories("Kaggle P100 GPU", categories=["user_facts"])
    assert len(hits) >= 1
    assert hits[0].id == created.id

    # Delete
    assert store2.delete_memory(created.id) is True
    assert store2.get_memory(created.id) is None


def test_memory_rejects_invalid_category(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "stan_memory.db")
    with pytest.raises(ValueError, match="Invalid memory category"):
        store.add_memory(category="invented_bucket", subject="x", content="y")
