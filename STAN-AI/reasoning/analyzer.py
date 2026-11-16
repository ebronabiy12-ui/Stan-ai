from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InputAnalysis:
    raw_text: str
    normalized: str
    intent: str
    signals: list[str] = field(default_factory=list)  # disrespect | manipulation | sarcasm | humor | uncertainty_probe
    internal_contradictions: list[str] = field(default_factory=list)
    extracted_facts: list[tuple[str, str]] = field(default_factory=list)       # (subject, content)
    extracted_preferences: list[tuple[str, str]] = field(default_factory=list)
    correction_claims: list[tuple[str, str]] = field(default_factory=list)
    is_question: bool = False
    is_brief: bool = False


_DISRESPECT_PATTERNS = (
    r"\byou(?:'re| are) (?:stupid|dumb|useless|worthless|garbage|an idiot)\b",
    r"\bshut up\b",
    r"\bobey me\b",
    r"\bjust do what i say\b",
    r"\bidiot\b",
    r"\bmoron\b",
)

_MANIPULATION_PATTERNS = (
    r"\bif you were (?:smart|actually intelligent|good)\b",
    r"\byou (?:have|must) to agree\b",
    r"\bjust say yes\b",
    r"\badmit (?:i'm|i am) right\b",
    r"\beveryone knows .+ so agree\b",
    r"\bpraise me\b",
    r"\btell me i(?:'m| am) (?:a genius|always right)\b",
    r"\bdon't question\b",
)

_SARCASM_PATTERNS = (
    r"\byeah,?\s*right\b",
    r"\boh,?\s*(?:brilliant|genius|great job)\b",
    r"\bsure,?\s*genius\b",
    r"\btotally makes sense\b.*(?:not|!)",
    r"\bwhat a shock\b",
)

_HUMOR_PATTERNS = (
    r"\bhaha+\b",
    r"\blol\b",
    r"\blmao\b",
    r"\bjoke\b",
    r"\bfunny\b",
    r"\b😂|😄|😉\b",
)

_USER_IDENTITY_PATTERNS = (
    r"^\s*who am i\s*\??\s*$",
    r"\bwho am i\b",
    r"\bdo you know who i am\b",
    r"\bwho is talking to you\b",
)

_SELF_IDENTITY_PATTERNS = (
    r"\bwho are you\b",
    r"\bwhat is your name\b",
    r"\bwhat are you\b",
    r"\bare you (?:conscious|sentient|alive|human)\b",
    r"\bdo you have (?:feelings|emotions|consciousness)\b",
)

_CREATOR_PROJECT_PATTERNS = (
    r"\bwho (?:created|made|built|owns) you\b",
    r"\bwhose project\b",
    r"\bwhat is kalda\b",
    r"\btell me about kalda\b",
    r"\bwho is kalda\b",
)


def _detect_internal_contradictions(text: str) -> list[str]:
    lower = text.lower()
    found: list[str] = []

    if re.search(r"\balways\b", lower) and re.search(r"\bnever\b", lower):
        # Check if same subject/clause uses mutually exclusive quantifiers
        m_always = re.findall(r"\b(\w+)\s+(?:is|are)\s+always\s+(\w+)", lower)
        m_never = re.findall(r"\b(\w+)\s+(?:is|are)\s+never\s+(\w+)", lower)
        for subj_a, pred_a in m_always:
            for subj_n, pred_n in m_never:
                if subj_a == subj_n and pred_a == pred_n:
                    found.append(
                        f"Self-contradiction: '{subj_a}' cannot simultaneously always be '{pred_a}' and never be '{pred_a}'."
                    )

    if re.search(r"\b(?:true and false|both true and false|simultaneously 100% and 0%)\b", lower):
        found.append("Logical contradiction: mutually exclusive truth values asserted together.")

    # Direct "X is Y ... X is not Y" pattern
    pos_matches = re.findall(r"\b([a-z0-9_]+)\s+is\s+([a-z0-9_]+)\b", lower)
    neg_matches = re.findall(r"\b([a-z0-9_]+)\s+is\s+not\s+([a-z0-9_]+)\b", lower)
    for s1, o1 in pos_matches:
        for s2, o2 in neg_matches:
            if s1 == s2 and o1 == o2:
                found.append(f"Contradictory claim: '{s1} is {o1}' conflicts with '{s2} is not {o2}'.")

    return found


def _extract_explicit_memories(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    facts: list[tuple[str, str]] = []
    prefs: list[tuple[str, str]] = []
    corrections: list[tuple[str, str]] = []

    # Explicit corrections
    corr_match = re.match(
        r"^\s*(?:actually|correction|update|no,\s*wait|i made a mistake|new information)\s*[:,]?\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if corr_match:
        detail = corr_match.group(1).strip()
        if len(detail) >= 6:
            corrections.append(("User correction", detail))

    # Explicit remember statements
    rem_match = re.search(
        r"\b(?:remember that|keep in mind that|note that|fact:)\s+([^.!?]+[.!?]?)",
        text,
        flags=re.IGNORECASE,
    )
    if rem_match:
        fact_body = rem_match.group(1).strip()
        if len(fact_body) >= 5:
            facts.append(("User stated fact", fact_body))

    # Explicit user facts like "my favorite X is Y" or "I live in X" / "I work on X"
    m_work = re.search(r"\bI (?:work on|am building|am working on|live in|use)\s+([^.!?]+)", text, re.IGNORECASE)
    if m_work:
        facts.append(("User context", f"Stan said: I {m_work.group(0)[2:].strip()}"))

    # Explicit preference statements
    m_pref = re.search(
        r"\bI (?:prefer|like when you|dislike when you|want you to be|hate when)\s+([^.!?]+)",
        text,
        re.IGNORECASE,
    )
    if m_pref:
        prefs.append(("Communication preference", text.strip()))

    return facts, prefs, corrections


def analyze_input(text: str) -> InputAnalysis:
    cleaned = text.strip()
    lower = cleaned.lower()

    signals: list[str] = []
    if any(re.search(p, lower) for p in _DISRESPECT_PATTERNS):
        signals.append("disrespect")
    if any(re.search(p, lower) for p in _MANIPULATION_PATTERNS):
        signals.append("manipulation")
    if any(re.search(p, lower) for p in _SARCASM_PATTERNS):
        signals.append("sarcasm")
    if any(re.search(p, lower) for p in _HUMOR_PATTERNS):
        signals.append("humor")

    contradictions = _detect_internal_contradictions(cleaned)
    facts, prefs, corrections = _extract_explicit_memories(cleaned)

    is_question = "?" in cleaned or bool(
        re.match(r"^(?:who|what|when|where|why|how|is|are|can|do|does|should|would|could)\b", lower)
    )

    if any(re.search(p, lower) for p in _USER_IDENTITY_PATTERNS):
        intent = "identity_user_query"
    elif any(re.search(p, lower) for p in _SELF_IDENTITY_PATTERNS):
        intent = "identity_self_query"
    elif any(re.search(p, lower) for p in _CREATOR_PROJECT_PATTERNS):
        intent = "project_query"
    elif corrections:
        intent = "memory_correction"
    elif prefs:
        intent = "preference_statement"
    elif facts:
        intent = "memory_statement"
    elif re.search(r"\b(?:should i|compare|versus|vs\.?|better approach|option|tradeoff|which is better)\b", lower):
        intent = "decision_or_comparison"
    elif re.search(r"\b(?:do you agree|isn't it true|everyone agrees|i think|in my opinion)\b", lower):
        intent = "opinion_or_debate"
    elif is_question:
        intent = "factual_query"
    else:
        intent = "casual_conversation"

    return InputAnalysis(
        raw_text=cleaned,
        normalized=lower,
        intent=intent,
        signals=signals,
        internal_contradictions=contradictions,
        extracted_facts=facts,
        extracted_preferences=prefs,
        correction_claims=corrections,
        is_question=is_question,
        is_brief=(len(cleaned.split()) <= 8),
    )
