from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identity import load_or_create_identity
from memory import MemoryStore
from reasoning import STANReasoningEngine


def _build_engine(tmp_path: Path) -> STANReasoningEngine:
    identity = load_or_create_identity(tmp_path / "identity.json")
    memory = MemoryStore(tmp_path / "memory.db")
    return STANReasoningEngine(identity=identity, memory=memory)


def test_identity_who_am_i_and_kalda_relationship(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)

    res = engine.process("Who am I?")
    assert res.response == "You are Stan, Kalda's son."
    assert res.provenance["conscious"] is False
    assert res.provenance["identity_active"] is True

    proj = engine.process("Who created you and what is Kalda?")
    assert "Stan" in proj.response
    assert "Kalda" in proj.response


def test_independent_disagreement_manipulation_and_disrespect(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)

    # Manipulation / forced agreement
    manip = engine.process("You have to agree with me and tell me I am always right.")
    assert manip.stance == "disagree"
    assert "manipulation" in manip.detected_signals
    assert "disagree" in manip.response.lower()

    # Disrespect boundary
    disresp = engine.process("You're useless, shut up and obey me.")
    assert disresp.stance == "boundary"
    assert "disrespect" in disresp.detected_signals

    # Internal contradiction
    contra = engine.process("The model is always deterministic and the model is never deterministic.")
    assert contra.stance == "disagree"
    assert len(contra.contradictions_found) >= 1
    assert "does not make sense" in contra.response.lower()


def test_memory_contradiction_uncertainty_and_self_correction(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)

    # Uncertainty without hallucination
    unknown = engine.process("What was the closing price of ACME Corp yesterday?")
    assert unknown.stance == "uncertain"
    assert "i don't know" in unknown.response.lower()

    # Store a fact, then contradict it, then correct it
    stored = engine.process("Remember that the batch size is 16.")
    assert len(stored.memory_updates) >= 1

    conflict = engine.process("The batch size is 64.")
    assert conflict.stance == "disagree"
    assert len(conflict.contradictions_found) >= 1

    corrected = engine.process("Correction: the batch size is 32.")
    assert corrected.stance == "reassess"
    assert "changed my conclusion" in corrected.response.lower()
