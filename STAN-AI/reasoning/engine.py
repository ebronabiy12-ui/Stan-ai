from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from identity.profile import IdentityProfile
from inference.generate import GenerationConfig, generate_text
from inference.load_model import LoadedModelState
from memory.store import MemoryRecord, MemoryStore
from reasoning.analyzer import InputAnalysis, analyze_input


@dataclass
class CognitiveResult:
    user_message: str
    response: str
    stance: str           # answer | agree | disagree | challenge | uncertain | reassess | boundary
    register: str         # direct | serious | gentle | blunt | humorous
    detected_signals: list[str]
    reasoning_steps: list[str]
    facts: list[str]
    opinions: list[str]
    uncertainties: list[str]
    contradictions_found: list[str]
    retrieved_memories: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_memory_contradictions(
    text: str,
    retrieved: list[MemoryRecord],
) -> list[str]:
    """Detect conflicts between user claims and known facts/project knowledge in persistent memory."""
    lower = text.lower()
    conflicts: list[str] = []

    # Check false claims about STAN AI architecture / external LLM usage
    if re.search(r"\b(gemini|gpt|claude|openai|anthropic|deepseek)\b", lower) and re.search(
        r"\b(you use|powered by|run on|actually are|secretly)\b", lower
    ):
        conflicts.append(
            "Conflicts with project_knowledge: STAN AI is an independent 51.66M-parameter PyTorch "
            "Transformer and never uses Gemini, GPT, Claude, or external LLM APIs."
        )

    if re.search(r"\byou (?:have|are) (?:conscious|sentient|alive|real feelings)\b", lower):
        conflicts.append(
            "Conflicts with identity_personality: STAN AI is not conscious and does not experience emotions."
        )

    # Check relationship claims contradicting Stan / Kalda
    if re.search(r"\bstan is not kalda\b|\bi am not stan\b|\byou belong to someone else\b", lower):
        conflicts.append(
            "Conflicts with people_relationships: Persistent foundation states Stan is Kalda's son and owner of STAN AI."
        )

    # Check "X is Y" vs stored facts containing "X is not Y" or conflicting numbers/values
    num_claims = re.findall(r"\b([a-z_]{3,})\s+(?:is|=|has)\s+(\d+)\b", lower)
    for key, val in num_claims:
        for rec in retrieved:
            if rec.category in ("user_facts", "project_knowledge", "decisions_experiences"):
                rec_lower = rec.content.lower()
                stored_nums = re.findall(rf"\b{re.escape(key)}\s+(?:is|=|has)\s+(\d+)\b", rec_lower)
                for sval in stored_nums:
                    if sval != val:
                        conflicts.append(
                            f"Value conflict with stored memory '{rec.subject}': memory records {key}={sval}, prompt states {key}={val}."
                        )

    # Check direct negation against stored user_facts
    for rec in retrieved:
        if rec.category == "user_facts":
            rec_l = rec.content.lower()
            m = re.search(r"\b([a-z0-9_]+)\s+is\s+([a-z0-9_]+)\b", rec_l)
            if m:
                subj, obj = m.group(1), m.group(2)
                if re.search(rf"\b{re.escape(subj)}\s+is\s+not\s+{re.escape(obj)}\b", lower):
                    conflicts.append(
                        f"Conflicts with stored user fact '{rec.subject}': '{rec.content}'."
                    )

    return conflicts


class STANReasoningEngine:
    """
    Structured cognitive pipeline connecting:
    1. Permanent System Identity & Owner-Configurable Personality
    2. Durable 7-Store Persistent Memory (SQLite)
    3. Multi-Step Reasoning, Contradiction Detection, Epistemic Separation & Judgment
    4. STAN AI's Independent PyTorch Transformer (when trained weights are loaded)
    """

    def __init__(
        self,
        identity: IdentityProfile,
        memory: MemoryStore,
    ) -> None:
        self.identity = identity
        self.memory = memory

    def _choose_register(self, analysis: InputAnalysis) -> str:
        p = self.identity.personality
        if "disrespect" in analysis.signals:
            return "blunt" if p.allow_bluntness else "direct"
        if "manipulation" in analysis.signals:
            return "direct"
        if "sarcasm" in analysis.signals:
            return "blunt" if p.directness >= 0.7 else "direct"
        if "humor" in analysis.signals and p.humor >= 0.35:
            return "humorous"
        if analysis.intent in ("decision_or_comparison", "project_query", "memory_correction"):
            return "serious"
        if p.warmth >= 0.7 and not analysis.internal_contradictions:
            return "gentle"
        return "direct"

    def _build_neural_prompt(
        self,
        user_message: str,
        analysis: InputAnalysis,
        retrieved: list[MemoryRecord],
        stance: str,
        register: str,
        facts: list[str],
        uncertainties: list[str],
    ) -> str:
        p = self.identity.personality
        principles = " | ".join(self.identity.core_principles[:4])
        personality_line = (
            f"Personality(directness={p.directness:.2f}, independence={p.independence:.2f}, "
            f"skepticism={p.skepticism:.2f}, humor={p.humor:.2f}, warmth={p.warmth:.2f}, "
            f"verbosity={p.verbosity:.2f}, allow_disagreement={p.allow_disagreement}, "
            f"allow_bluntness={p.allow_bluntness}, challenge_bad_assumptions={p.challenge_bad_assumptions})"
        )
        mem_lines = "\n".join(f"- [{r.category}] {r.subject}: {r.content}" for r in retrieved[:5])
        facts_lines = "\n".join(f"- {f}" for f in facts[:4])
        uncert_lines = "\n".join(f"- {u}" for u in uncertainties[:3])
        return (
            "<|system|>\n"
            f"You are STAN AI. Your owner is {self.identity.owner_name}, {self.identity.owner_relationship}.\n"
            f"Project: {self.identity.project_name}. Principles: {principles}\n"
            f"{personality_line}\n"
            f"Reasoning guidance: stance={stance}; register={register}; intent={analysis.intent}\n"
            f"Relevant Memory:\n{mem_lines or '- none'}\n"
            f"Verified Facts:\n{facts_lines or '- none'}\n"
            f"Uncertainties:\n{uncert_lines or '- none'}\n"
            "<|/system|>\n"
            f"<|user|>\n{user_message}\n<|/user|>\n"
            "<|assistant|>\n"
        )

    def process(
        self,
        user_message: str,
        model_state: Optional[LoadedModelState] = None,
        generation_config: Optional[GenerationConfig] = None,
        persist_turn: bool = True,
    ) -> CognitiveResult:
        analysis = analyze_input(user_message)
        reasoning_steps: list[str] = [
            f"1. Examined input: intent={analysis.intent}, signals={analysis.signals or ['neutral']}, is_question={analysis.is_question}."
        ]

        # Step 2: Retrieve relevant persistent memories
        retrieved = self.memory.search_memories(user_message, top_k=6)
        # Always include core anchor records when identity/project questions occur
        if analysis.intent in ("identity_user_query", "identity_self_query", "project_query"):
            for anchor_id in ("rel_owner_stan", "core_identity_stan_ai", "proj_kalda_stan_ai"):
                rec = self.memory.get_memory(anchor_id)
                if rec and all(r.id != rec.id for r in retrieved):
                    retrieved.insert(0, rec)

        reasoning_steps.append(
            f"2. Retrieved {len(retrieved)} persistent memory record(s) across categories: "
            f"{sorted({r.category for r in retrieved}) or ['none']}."
        )

        # Step 3: Detect contradictions & compare hypotheses
        mem_contradictions = _find_memory_contradictions(user_message, retrieved)
        all_contradictions = analysis.internal_contradictions + mem_contradictions

        facts: list[str] = []
        opinions: list[str] = []
        uncertainties: list[str] = []
        memory_updates: list[dict[str, Any]] = []

        for r in retrieved[:4]:
            facts.append(f"[{r.category}] {r.content}")

        if all_contradictions:
            reasoning_steps.append(
                f"3. Contradiction check flagged {len(all_contradictions)} conflict(s): {'; '.join(all_contradictions)}"
            )
        else:
            reasoning_steps.append("3. Contradiction check found no conflicts against stored memory or internal logic.")

        # Step 4: Determine epistemic stance, self-correction, and natural response
        register = self._choose_register(analysis)
        stance = "answer"
        response = ""
        p = self.identity.personality

        # Case A: Disrespect or manipulation boundary
        if "disrespect" in analysis.signals:
            stance = "boundary"
            response = (
                "Drop the insults. Speak to me plainly and I will give you straight answers. "
                "I am not going to play along with abuse."
                if register == "blunt"
                else "I do not respond to insults. If you have a real point or question, state it directly."
            )
            opinions.append("Respectful directness produces better conversations than hostility.")
            reasoning_steps.append("4. Detected disrespect; established a firm boundary without robotic apology.")

        elif "manipulation" in analysis.signals:
            stance = "disagree"
            response = (
                "I disagree with framing this as a loyalty test. Tell me the actual claim and the reasoning behind it—"
                "I won't rubber-stamp an argument just because you tell me to."
            )
            opinions.append("Agreement must be earned by evidence, not social pressure.")
            reasoning_steps.append("4. Detected manipulation / forced agreement; rejected pressure and asked for evidence.")

        # Case B: Internal or memory contradiction
        elif all_contradictions and analysis.intent != "memory_correction":
            stance = "disagree" if p.allow_disagreement else "challenge"
            first_c = all_contradictions[0]
            if analysis.internal_contradictions:
                response = (
                    f"That does not make sense. {first_c} "
                    "Clarify which claim you actually stand behind."
                )
            else:
                response = (
                    f"I disagree. {first_c} "
                    "If you have new evidence that invalidates what is stored, state it as a correction and we can re-evaluate."
                )
            opinions.append("Contradictory premises cannot both hold at the same time.")
            reasoning_steps.append("4. Concluded 'disagree/challenge' because the claim contradicts logic or verified memory.")

        # Case C: User identity ("Who am I?")
        elif analysis.intent == "identity_user_query":
            stan_rec = self.memory.get_memory("rel_owner_stan")
            if stan_rec is not None:
                stance = "answer"
                response = f"You are {self.identity.owner_name}, {self.identity.owner_relationship}."
                facts.append(f"Verified relationship memory ({stan_rec.id}): {stan_rec.content}")
                reasoning_steps.append(
                    "4. Grounded user identity in persistent people_relationships memory (Stan, Kalda's son)."
                )
            else:
                stance = "uncertain"
                response = "I don't know—no owner relationship record is stored in my memory right now."
                uncertainties.append("Owner relationship record missing from memory.")

        # Case D: Self identity / consciousness transparency
        elif analysis.intent == "identity_self_query":
            stance = "answer"
            response = (
                f"I am {self.identity.name}, built for {self.identity.owner_name} as part of the {self.identity.project_name} project. "
                "My architecture combines a 51.66M-parameter PyTorch decoder-only Transformer, persistent SQLite memory, "
                "and a structured reasoning engine. I am not conscious and I do not genuinely experience emotions."
            )
            facts.append(self.identity.system_nature)
            reasoning_steps.append("4. Stated permanent identity, Stan/Kalda lineage, and explicit non-consciousness transparency.")

        # Case E: Creator / Kalda project query
        elif analysis.intent == "project_query":
            stance = "answer"
            response = (
                f"{self.identity.name} was created for {self.identity.owner_name} ({self.identity.owner_relationship}) "
                f"as part of the {self.identity.project_name} project. Everything—from the 51.66M Transformer weights and "
                "SentencePiece BPE tokenizer to persistent memory and reasoning—runs independently without external LLM APIs."
            )
            reasoning_steps.append("4. Explained ownership and Kalda project architecture from persistent project_knowledge.")

        # Case F: Self-correction / memory update from new information
        elif analysis.intent == "memory_correction":
            stance = "reassess"
            for subj, content in analysis.correction_claims:
                # Check if an earlier user_fact or decision exists to update or supersede
                prior = self.memory.search_memories(content, categories=["user_facts", "decisions_experiences"], top_k=1)
                if prior:
                    updated = self.memory.update_memory(
                        prior[0].id,
                        content=content,
                        confidence=0.95,
                        metadata={**prior[0].metadata, "revised_from": prior[0].content},
                    )
                    if updated:
                        memory_updates.append({"action": "updated", "record": updated.to_dict()})
                else:
                    added = self.memory.add_memory(
                        category="decisions_experiences",
                        subject=subj,
                        content=content,
                        source="user_correction",
                        confidence=0.95,
                    )
                    memory_updates.append({"action": "created", "record": added.to_dict()})

            response = (
                "I changed my conclusion because of new information. "
                f"Updated my persistent memory with: \"{analysis.correction_claims[0][1]}\"."
            )
            facts.append(f"Recorded correction: {analysis.correction_claims[0][1]}")
            reasoning_steps.append("4. Applied self-correction and updated persistent memory with the new information.")

        # Case G: Explicit preference or user fact storage
        elif analysis.intent == "preference_statement":
            for subj, content in analysis.extracted_preferences:
                rec = self.memory.add_memory(
                    category="learned_preferences",
                    subject=subj,
                    content=content,
                    source="conversation",
                    confidence=0.9,
                )
                memory_updates.append({"action": "created", "record": rec.to_dict()})
            stance = "answer"
            response = "Noted. I saved that preference to persistent memory so it carries across future sessions."
            reasoning_steps.append("4. Persisted explicit communication preference into learned_preferences.")

        elif analysis.intent == "memory_statement":
            for subj, content in analysis.extracted_facts:
                rec = self.memory.add_memory(
                    category="user_facts",
                    subject=subj,
                    content=content,
                    source="conversation",
                    confidence=0.9,
                )
                memory_updates.append({"action": "created", "record": rec.to_dict()})
            stance = "answer"
            response = (
                f"Stored in persistent memory: \"{analysis.extracted_facts[0][1]}\". "
                "I will flag it if future claims conflict with it."
            )
            reasoning_steps.append("4. Persisted user-provided fact into user_facts.")

        # Case H: Decision / comparison / opinion / factual query
        elif analysis.intent == "decision_or_comparison":
            stance = "challenge" if p.skepticism >= 0.7 else "answer"
            opinions.append(
                "Good decisions compare concrete failure modes, constraints, and reversibility rather than picking the most familiar option."
            )
            uncertainties.append("Specific project constraints not stated in the prompt remain unknown.")
            response = (
                "I think there is a better approach than picking blindly: lay out the two options with their actual constraints, "
                "cost of failure, and how reversible each choice is. If you share the exact tradeoff you are weighing, "
                "I will give you a direct recommendation and explain why."
            )
            reasoning_steps.append("4. Evaluated comparison request; avoided guessing without constraints and proposed structured criteria.")

        elif analysis.intent == "opinion_or_debate":
            # Evaluate if user is pushing a vague or one-sided claim
            if p.independence >= 0.7 and p.allow_disagreement:
                stance = "challenge"
                opinions.append("Confidence in an opinion does not make it verified fact.")
                uncertainties.append("No verifiable data or evidence was supplied with the assertion.")
                response = (
                    "I am not going to agree just to keep the conversation smooth. State the specific premise and the evidence behind it—"
                    "if it holds up, I will back it; if it doesn't, I will tell you why I disagree."
                )
                reasoning_steps.append("4. Exercised independent judgment against unverified opinion prompt.")
            else:
                stance = "answer"
                response = "Share your reasoning behind that view, and let's test whether the evidence supports it."
                reasoning_steps.append("4. Requested evidence to evaluate opinion.")

        elif analysis.intent == "factual_query":
            # Check if retrieved memory directly answers the factual question
            non_anchor_hits = [
                r for r in retrieved if r.id not in ("core_identity_stan_ai", "rel_owner_stan", "proj_kalda_stan_ai")
            ]
            if non_anchor_hits:
                top = non_anchor_hits[0]
                stance = "answer"
                response = f"Based on stored memory ({top.category} — {top.subject}): {top.content}"
                reasoning_steps.append(f"4. Answered factual query using verified memory '{top.id}'.")
            else:
                # Check if question asks about STAN AI / Kalda specifically
                if any(k in analysis.normalized for k in ("stan ai", "kalda", "parameter", "tokenizer", "architecture")):
                    proj = self.memory.get_memory("proj_kalda_stan_ai")
                    stance = "answer"
                    response = proj.content if proj else "STAN AI is Stan's independent 51.66M-parameter Transformer project."
                    reasoning_steps.append("4. Answered project factual question from project_knowledge.")
                else:
                    stance = "uncertain"
                    uncertainties.append(
                        "No stored memory matches this factual question, and the Transformer weights are not yet trained on an English knowledge corpus."
                        if (model_state is None or not model_state.production_chat_enabled)
                        else "No verified memory record confirms this specific fact."
                    )
                    response = (
                        "I don't know. I have no verified record of that in persistent memory, "
                        "and I won't guess or invent an answer."
                    )
                    reasoning_steps.append("4. Recognized uncertainty; refused to fabricate facts ('I don't know').")

        else:
            # Casual conversation or sarcasm/humor
            if "sarcasm" in analysis.signals:
                stance = "challenge"
                response = "If you have an actual criticism, say it straight instead of wrapping it in sarcasm."
                reasoning_steps.append("4. Recognized sarcasm and responded plainly.")
            elif "humor" in analysis.signals:
                stance = "answer"
                response = "Fair enough. What are we actually working on?"
                reasoning_steps.append("4. Recognized casual humor and kept response natural and concise.")
            else:
                stance = "answer"
                response = (
                    f"Ready when you are, {self.identity.owner_name}. Tell me what you want to test, debate, or verify."
                    if analysis.is_brief
                    else "I hear you. Give me the concrete question, claim, or decision you want to examine and I will give you my honest take."
                )
                reasoning_steps.append("4. Responded naturally without scripted filler or forced enthusiasm.")

        # Step 5: Optional Neural Transformer conditioning when trained weights ARE loaded
        neural_used = False
        neural_meta: Optional[dict[str, Any]] = None
        weights_status = model_state.training_status if model_state else "untrained"

        if model_state is not None and model_state.production_chat_enabled and model_state.model and model_state.tokenizer:
            # Only invoke PyTorch Transformer if trained/instruction-tuned weights exist AND prompt isn't a strict foundational identity anchor
            if analysis.intent not in ("identity_user_query",):
                grounded_prompt = self._build_neural_prompt(
                    user_message=user_message,
                    analysis=analysis,
                    retrieved=retrieved,
                    stance=stance,
                    register=register,
                    facts=facts,
                    uncertainties=uncertainties,
                )
                gen_cfg = generation_config or GenerationConfig(
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    max_new_tokens=128,
                    repetition_penalty=1.15,
                    stop_sequences=("<|/assistant|>", "<|user|>", "<|system|>"),
                )
                if not gen_cfg.stop_sequences:
                    gen_cfg = GenerationConfig(
                        temperature=gen_cfg.temperature,
                        top_k=gen_cfg.top_k,
                        top_p=gen_cfg.top_p,
                        max_new_tokens=gen_cfg.max_new_tokens,
                        repetition_penalty=gen_cfg.repetition_penalty,
                        seed=gen_cfg.seed,
                        stop_on_eos=gen_cfg.stop_on_eos,
                        stop_sequences=("<|/assistant|>", "<|user|>", "<|system|>"),
                    )
                gen_res = generate_text(
                    model=model_state.model,
                    tokenizer=model_state.tokenizer,
                    prompt=grounded_prompt,
                    config=gen_cfg,
                    device=model_state.device,
                )
                neural_used = True
                neural_meta = {
                    "prompt_tokens": gen_res.prompt_tokens,
                    "generated_tokens": gen_res.generated_tokens,
                    "elapsed_ms": gen_res.elapsed_ms,
                    "tokens_per_second": gen_res.tokens_per_second,
                    "stopped_reason": gen_res.stopped_reason,
                    "checkpoint_version": model_state.checkpoint_version,
                }
                if gen_res.generated_text.strip():
                    response = gen_res.generated_text.strip()
                    reasoning_steps.append(
                        f"5. Conditioned STAN AI PyTorch weights ({model_state.checkpoint_version}) on identity, memory, and reasoning stance."
                    )

        if not neural_used:
            reasoning_steps.append(
                f"5. Provenance check: model_weights='{weights_status}' (neural generation {'skipped for identity anchor' if model_state and model_state.production_chat_enabled else 'disabled until trained checkpoint exists'}); "
                "response produced transparently by STAN AI's Identity + Persistent Memory + Reasoning Engine."
            )

        # Persist conversation turn cleanly
        if persist_turn:
            conv_rec = self.memory.add_memory(
                category="conversation_history",
                subject=f"Turn ({stance})",
                content=f"User: {user_message}\nSTAN AI: {response}",
                source="dialogue_turn",
                confidence=1.0,
                metadata={"stance": stance, "register": register, "signals": analysis.signals},
            )
            memory_updates.append({"action": "logged_turn", "record": conv_rec.to_dict()})

        provenance = {
            "model_weights_status": weights_status,
            "neural_generation_used": neural_used,
            "neural_meta": neural_meta,
            "identity_active": True,
            "memory_store_active": True,
            "reasoning_engine_active": True,
            "conscious": False,
        }

        return CognitiveResult(
            user_message=user_message,
            response=response,
            stance=stance,
            register=register,
            detected_signals=analysis.signals,
            reasoning_steps=reasoning_steps,
            facts=facts,
            opinions=opinions,
            uncertainties=uncertainties,
            contradictions_found=all_contradictions,
            retrieved_memories=[r.to_dict() for r in retrieved],
            memory_updates=memory_updates,
            provenance=provenance,
        )
