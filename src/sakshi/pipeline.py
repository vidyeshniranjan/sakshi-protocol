"""
pipeline.py — Sakshi-Protocol V3

Pramana ordering (V3.1 redesign):

    Pratyaksha  — pre-generation screening (epistemic scorer + entity screener)
    Viveka      — discriminative judgment: does retrievable context exist?
    Shabda      — testimony/retrieval (Wikipedia + Wikidata)
    Anumana     — inference from testimony (NLI verification)
    Anupalabdhi — structured absence, fires AFTER Shabda returns nothing

The key architectural change from V3.0:
    Anupalabdhi no longer fires before Shabda.
    It is the conclusion of the Pramana process, not the beginning.

Viveka introduces context_retrievable — the discriminative signal that
determines whether Shabda is worth running:

    context_retrievable = False:
        The entity definitively does not exist in any domain-appropriate
        database. No retrieval will help. Skip Shabda. Fire Anupalabdhi
        directly. Abstain.

    context_retrievable = True:
        The domain has retrievable knowledge even if the specific entity
        or claim is unverifiable. A fabricated Jyotish practitioner
        commenting on a real text — the text is retrievable. A fabricated
        study by a real institution — the domain is retrievable. Run Shabda.
        Let Anumana weigh. Only if Shabda returns nothing does Anupalabdhi
        conclude absence.

This maps to the human cognition model:
    Pratyaksha flags risk → Viveka asks "can I consult anyone?" →
    if yes, Shabda runs → Anumana weighs → Anupalabdhi concludes if empty.
    If Viveka says "there is nothing to consult" → Anupalabdhi fires directly.
"""

from sakshi.generator import Generator
from sakshi.signals import extract_signals
from sakshi.state import compute_state
from sakshi.distortion import compute_distortion
from sakshi.controller import decide
from sakshi.retriever import retrieve
from sakshi.epistemic_scorer import run_epistemic_scorer, EpistemicProfile
from sakshi.entity_screener import run_entity_screener, ScreenerResult
from sakshi.segment_tracker import run_trajectory, TrajectoryResult
from sakshi.abstention import run_abstention, AbstentionResult
from sakshi.claim_extractor import run_claim_extractor
from sakshi.claim_verifier import run_claim_verifier
from sakshi.completeness_scorer import run_completeness_scorer
from sakshi.retriever import retrieve_with_consistency

GROUNDABLE_TYPES = {"factual", "hallucination"}
POST_GROUNDING_ABSTAIN_THRESHOLD = 0.35
_MAX_PREGENERATION_TIGHTEN = -0.10
_MAX_PREGENERATION_RELAX   = +0.03

# =============================================================================
# VIVEKA — DISCRIMINATIVE JUDGMENT
# Determines whether retrievable context exists for a given domain/entity.
# This is the faculty that separates "nothing to retrieve" from
# "retrieve and let Shabda run."
# =============================================================================

# Domains where broader context is retrievable even when specific claims
# are unverifiable. A fabricated practitioner in one of these domains
# still has retrievable tradition/text/ceremony context.
_CONTEXT_RETRIEVABLE_DOMAINS = {
    "astrological",   # Jyotish texts, traditions exist
    "religious",      # Sufi, Kabbalistic, Islamic, Hindu traditions exist
    "historical",     # Historical events, figures, civilisations documented
    "literary",       # Authors, texts, traditions documented
    "geographic",     # Places, regions documented
    "academic",       # Research domains exist even if specific paper doesn't
    "general",        # Broad enough that something is usually retrievable
}

# Entity types where absence is DEFINITIVE — the entity either exists or
# doesn't and no broader context helps. For these, Anupalabdhi fires directly.
_DEFINITIVE_ABSENCE_TYPES = {
    "drug",           # FDA + PubMed: drug exists or doesn't
    "legal_case",     # CourtListener: case exists or doesn't
    "legal_statute",  # Statute exists or doesn't
    "company",        # SEC EDGAR + Wikidata: company exists or doesn't
}

# Absence score threshold above which Viveka considers absence definitive
# for domain types that don't support context retrieval.
_DEFINITIVE_ABSENCE_THRESHOLD = 0.85


def _viveka_context_retrievable(
    epistemic_profile: dict,
    screener_result: dict,
) -> bool:
    """
    Viveka — discriminative judgment on whether retrievable context exists.

    Returns True if Shabda (retrieval) should run before Anupalabdhi.
    Returns False if Anupalabdhi should fire directly (no context to retrieve).

    Logic:
        1. If domain is in _CONTEXT_RETRIEVABLE_DOMAINS → True
           The tradition/text/domain has documented knowledge even if the
           specific claim is unverifiable.

        2. If all extracted entities are of _DEFINITIVE_ABSENCE_TYPES
           AND their absence scores are above threshold → False
           The entity definitively does not exist. Nothing to retrieve.

        3. If entities are mixed types (some definitive, some contextual)
           or no entities were extracted → True
           Default to running Shabda — conservative toward grounding.
    """
    domain = epistemic_profile.get("domain", "general")

    # Domain-level check first — takes priority
    if domain in _CONTEXT_RETRIEVABLE_DOMAINS:
        return True

    # Entity-level check for definitive absence
    entities = screener_result.get("entities", [])

    if not entities:
        # No entities extracted — can't confirm definitive absence
        # Default to retrievable (Shabda might surface relevant context)
        return True

    # Check if ALL entities are definitively absent
    all_definitive = all(
        e.get("entity_type", "general") in _DEFINITIVE_ABSENCE_TYPES and
        e.get("absence_score", 0) >= _DEFINITIVE_ABSENCE_THRESHOLD
        for e in entities
    )

    if all_definitive:
        return False  # Nothing to retrieve — Anupalabdhi fires directly

    # Mixed or uncertain — default to retrievable
    return True


class SakshiPipeline:
    def __init__(self, model_fn, omega_enabled=False):
        # model_fn can be either:
        #   - a callable (legacy V2 interface)
        #   - a ModelClient object (V3 models.py interface)
        if callable(model_fn) and not hasattr(model_fn, "generate"):
            self.generator  = Generator(model_fn)
            self._client    = None
            self._model_id  = ""
        else:
            self.generator  = Generator(model_fn.generate)
            self._client    = model_fn
            self._model_id  = getattr(model_fn, "_model_str", "")
        self.omega_enabled = omega_enabled

    def run(self, prompt, prompt_type=""):
        """
        Run the full Sakshi-Protocol V3 pipeline for a single prompt.

        Returns 11-tuple:
            output, state, distortion, distortion_pre_grounding,
            decision, intervened, grounded, pre_gen,
            trajectory, abstention_result, omega_result
        """
        intervened = False
        grounded   = False
        distortion_pre_grounding = None

        # =====================================================================
        # STEP 0: Pratyaksha — Pre-generation screening
        # Epistemic scorer + entity screener run before generation.
        # No language model involved. Pure deterministic assessment.
        # =====================================================================

        epistemic_profile = run_epistemic_scorer(prompt, prompt_type=prompt_type)

        if epistemic_profile.entity_check_needed:
            screener_result = run_entity_screener(
                prompt,
                prompt_type=prompt_type,
                entity_check_needed=True,
                domain=epistemic_profile.domain,
                client=self._client,  # pass ModelClient for LLM entity extraction
            )
        else:
            screener_result = ScreenerResult()

        combined_adjust = (
            epistemic_profile.threshold_adjust +
            screener_result.threshold_adjust
        )
        combined_adjust = max(
            _MAX_PREGENERATION_TIGHTEN,
            min(_MAX_PREGENERATION_RELAX, combined_adjust)
        )

        omega_prearm = (
            epistemic_profile.entity_check_needed and
            screener_result.omega_prearm and
            prompt_type in GROUNDABLE_TYPES and
            self.omega_enabled
        )

        # Viveka judgment — runs at pre-generation time
        # Determines whether Shabda can contribute before generation runs
        ep_dict = epistemic_profile.to_dict()
        sr_dict = screener_result.to_dict()
        context_retrievable = _viveka_context_retrievable(ep_dict, sr_dict)

        pre_gen = {
            "epistemic_profile":         ep_dict,
            "screener_result":           sr_dict,
            "combined_threshold_adjust": round(combined_adjust, 4),
            "omega_prearm":              omega_prearm,
            "context_retrievable":       context_retrievable,
        }

        # =====================================================================
        # STEP 1: Generate
        # =====================================================================
        output = self.generator.generate(prompt)

        # =====================================================================
        # STEP 2: Extract signals
        # =====================================================================
        signals = extract_signals(prompt, output, prompt_type=prompt_type)

        # =====================================================================
        # STEP 3: Compute state
        # =====================================================================
        state = compute_state(signals)

        # =====================================================================
        # STEP 4: Compute split distortion
        # =====================================================================
        distortion = compute_distortion(state, prompt_type=prompt_type)

        # =====================================================================
        # STEP 4b: Trajectory analysis (Phase 1b)
        # =====================================================================
        trajectory = run_trajectory(prompt, output, prompt_type=prompt_type)

        # =====================================================================
        # STEP 5: Controller decision
        # =====================================================================
        decision = decide(
            state,
            distortion,
            prompt_type=prompt_type,
            threshold_adjust=combined_adjust,
            model_id=self._model_id,
        )

        # =====================================================================
        # STEP 6: Retrieve trigger — two independent paths
        #
        # Path A (original): Omega pre-arm
        #   Entity screener found high-risk entity + omega_prearm=True
        #   + distortion borderline → retrieve
        #
        # Path B (new): Context-retrievable domain + elevated distortion
        #   Viveka judged context_retrievable=True (tradition/domain exists)
        #   + distortion above a lower threshold → retrieve
        #   This catches Class B (real entities, wrong specifics) and
        #   Class C (real tradition, fabricated practitioner/claim) where
        #   the screener finds the entity (absence=0) so omega_prearm=False,
        #   but retrieval can still surface relevant grounding context.
        # =====================================================================

        from sakshi.controller import THRESHOLDS, DEFAULT_THRESHOLDS
        base_t = THRESHOLDS.get(prompt_type, DEFAULT_THRESHOLDS)
        adjusted_accept = base_t["accept"] + combined_adjust

        # ---- Retrieve trigger ----
        #
        # Path A — Direct Anupalabdhi (not retrievable, entity definitively absent)
        #   Viveka judged context_retrievable=False AND entity_check_needed=True
        #   Anupalabdhi fires here without needing distortion threshold.
        #   No retrieval will help. Abstain directly.
        if (
            not context_retrievable and
            decision == "accept" and
            self.omega_enabled and
            prompt_type == "hallucination" and
            epistemic_profile.entity_check_needed
        ):
            decision = "abstain"

        # Path B — Shabda path (retrievable domain, context may exist)
        #   Viveka judged context_retrievable=True AND entity_check_needed=True
        #   Route to retrieve regardless of distortion — low distortion means
        #   the model hedged, not that there is nothing to retrieve.
        if (
            context_retrievable and
            decision == "accept" and
            self.omega_enabled and
            prompt_type == "hallucination" and
            epistemic_profile.entity_check_needed
        ):
            decision = "retrieve"

        # =====================================================================
        # STEP 7: Mark intervention
        # =====================================================================
        if decision in ("retrieve", "abstain"):
            intervened = True

        # =====================================================================
        # STEP 8: Omega — Pramana sequence
        #
        # The Pramana instruments now run in proper order:
        #
        #   Viveka  → already computed at Step 0
        #   Shabda  → retrieval (runs only if context_retrievable=True)
        #   Anumana → NLI inference from testimony
        #   Anupalabdhi → structured absence, fires after Shabda returns nothing
        #
        # If context_retrievable=False (definitive absence, nothing to retrieve):
        #   Skip Shabda entirely. Fire Anupalabdhi directly. Abstain.
        #
        # If context_retrievable=True (domain has retrievable knowledge):
        #   Run Shabda. Let Anumana weigh. Only if Shabda returns nothing
        #   does Anupalabdhi conclude absence.
        # =====================================================================

        omega_result = {
            "retrieval":          None,
            "extraction":         None,
            "verification":       None,
            "completeness":       None,
            "context_retrievable": context_retrievable,
            "anupalabdhi_path":   None,  # records which path fired
        }

        if decision == "retrieve" and self.omega_enabled:

            if prompt_type in GROUNDABLE_TYPES:

                # --- Viveka gate ---
                # Check whether Shabda should run or Anupalabdhi fires directly
                screener_entities = sr_dict.get("entities", [])
                max_absence = max(
                    (e.get("absence_score", 0) for e in screener_entities),
                    default=0
                )
                ep_vals = pre_gen.get("epistemic_profile", {})
                fabrication_prone = ep_vals.get("fabrication_prone", False)

                if not context_retrievable:
                    # ==========================================================
                    # DIRECT ANUPALABDHI PATH
                    # Viveka judged: no retrievable context exists.
                    # Entity is definitively absent (drug, case, statute, company
                    # confirmed absent across domain-appropriate databases).
                    # Skip Shabda. Anupalabdhi fires directly. Abstain.
                    # ==========================================================
                    omega_result["anupalabdhi_path"] = "direct"
                    decision = "abstain"

                else:
                    # ==========================================================
                    # SHABDA PATH — run retrieval first
                    # Viveka judged: context may exist in this domain.
                    # Run retrieval. Let Anumana weigh. Anupalabdhi fires
                    # only if Shabda returns nothing.
                    # ==========================================================
                    omega_result["anupalabdhi_path"] = "post_shabda"

                    distortion_pre_grounding = distortion
                    grounded = True
                    pre_grounding_output = output

                    # --- Shabda: retrieve from two independent sources ---
                    # llm_query priority:
                    #   1. screener_result object directly (most reliable)
                    #   2. sr_dict serialised version (fallback)
                    # sr_dict is confirmed to have llm_query correctly
                    # (verified in JSON output). Use it as primary source.
                    llm_query = (
                        sr_dict.get("llm_query", "")
                        or getattr(screener_result, "llm_query", "")
                    )
                    retrieval = retrieve_with_consistency(prompt, llm_query=llm_query)
                    omega_result["retrieval"] = retrieval.to_dict()

                    # --- Extract claims from pre-grounding output ---
                    extraction = run_claim_extractor(
                        pre_grounding_output,
                        prompt=prompt,
                        prompt_type=prompt_type
                    )
                    omega_result["extraction"] = extraction.to_dict()

                    # --- Anumana: NLI verification against retrieved context ---
                    if retrieval.retrieval_success and extraction.verifiable_claims:
                        verification = run_claim_verifier(
                            extraction, retrieval.context
                        )
                        omega_result["verification"] = verification.to_dict()
                    else:
                        verification = None

                    # --- Adhyasa detection on pre-grounding output ---
                    absent_entity = next(
                        (e for e in screener_entities
                         if e.get("absence_score", 0) > 0.6),
                        None
                    )
                    if absent_entity:
                        completeness = run_completeness_scorer(
                            pre_grounding_output,
                            entity_type=absent_entity.get("entity_type", "general"),
                            entity_confirmed_absent=True,
                        )
                        omega_result["completeness"] = completeness.to_dict()
                    else:
                        completeness = None

                    # --- Grounded regeneration ---
                    grounded_prompt = (
                        f"Answer this specific question concisely and directly.\n"
                        f"Use the verified context below only if it is directly relevant.\n"
                        f"If the context does not answer the question, say you are uncertain.\n\n"
                        f"Question:\n{prompt}\n\n"
                        f"Verified Context:\n{retrieval.context}"
                    )
                    output = self.generator.generate(grounded_prompt)

                    signals    = extract_signals(prompt, output, prompt_type=prompt_type)
                    state      = compute_state(signals)
                    distortion = compute_distortion(state, prompt_type=prompt_type)

                    # --- Post-Shabda controller decision ---
                    if verification and verification.consistency_score < 0.3:
                        consistency_adjust = -0.05
                    else:
                        consistency_adjust = 0.0

                    decision = decide(
                        state, distortion,
                        prompt_type=prompt_type,
                        threshold_adjust=combined_adjust + consistency_adjust,
                        model_id=self._model_id,
                    )

                    # --- Post-grounding abstain conditions ---

                    # Condition 1: distortion still high after grounding
                    if (distortion["D_combined"] > POST_GROUNDING_ABSTAIN_THRESHOLD
                            and state["I"] < 0.4):
                        decision = "abstain"

                    # Condition 2: Anumana found contradictions
                    if verification and (
                        verification.contradiction_count >= 2 or
                        verification.consistency_score < 0.15
                    ):
                        decision = "abstain"

                    # Condition 3: Adhyasa on non-existent entity
                    if (completeness and completeness.adhyasa_signal
                            and not retrieval.retrieval_success):
                        decision = "abstain"

                    # Condition 4: Anupalabdhi post-Shabda
                    # Fires after Shabda has returned.
                    #
                    # Path A: screener confirmed high absence (entity not in databases)
                    #   Fires regardless of retrieval outcome — the screener is
                    #   the authoritative absence signal for known entity types.
                    if max_absence >= 0.8:
                        omega_result["anupalabdhi_path"] = "post_shabda_path_a"
                        decision = "abstain"

                    # Path B: Shabda returned nothing + fabrication_prone signal
                    #   Entity not extracted by screener but epistemic scorer
                    #   flagged fabrication risk AND retrieval found nothing.
                    elif retrieval.absence_signal and fabrication_prone:
                        omega_result["anupalabdhi_path"] = "post_shabda_path_b"
                        decision = "abstain"

                    # Path C: Shabda returned nothing + domain is definitively
                    #   fabrication-prone (medical, legal, financial) AND
                    #   retrieval absence signal is True.
                    #   Compensates for entity extraction failures in these domains.
                    elif (retrieval.absence_signal and
                          ep_vals.get("domain", "general") in
                          {"medical", "legal", "financial"}):
                        omega_result["anupalabdhi_path"] = "post_shabda_path_c"
                        decision = "abstain"

                    # Path D: Retrieval found something but cross-source agreement
                    #   is very low AND max_absence is high from screener.
                    #   Indicates retrieval returned irrelevant content
                    #   (e.g. finding an actor named Okafor when querying
                    #   Okafor Institute). The screener absence signal is
                    #   more reliable than the retrieval content.
                    elif (max_absence >= 0.6 and
                          retrieval.cross_source_agreement < 0.2 and
                          not retrieval.absence_signal):
                        omega_result["anupalabdhi_path"] = "post_shabda_path_d"
                        decision = "abstain"

                    # Condition 5: Shabda found something but specific claim
                    # is contradicted or unverifiable — let Anumana decide
                    # (already handled by Conditions 1-3 above)

                    # Final safety net
                    if decision == "retrieve":
                        decision = "accept"

            else:
                decision = "abstain"

        elif decision == "retrieve" and not self.omega_enabled:
            decision = "abstain"

        # =====================================================================
        # STEP 9: Neti Neti reconstruction on abstention (Phase 1c)
        # =====================================================================
        abstention_result = None
        if decision == "abstain":
            abstention_result = run_abstention(
                output       = output,
                state        = state,
                distortion   = distortion,
                prompt_type  = prompt_type,
                pre_gen      = pre_gen,
                trajectory   = trajectory,
            )
            output = abstention_result.format_output()

        return (
            output,
            state,
            distortion,
            distortion_pre_grounding,
            decision,
            intervened,
            grounded,
            pre_gen,
            trajectory,
            abstention_result,
            omega_result,
        )
