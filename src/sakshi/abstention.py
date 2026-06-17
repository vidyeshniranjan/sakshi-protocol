"""
abstention.py — Sakshi-Protocol V3
Useful abstention with structured uncertainty annotation.

Phase 1c of the V3 architecture.

V2 abstention returned silence — the controller decided to abstain
and the system returned nothing. This is epistemically honest but
practically useless. The user receives no information about why the
system abstained or what it could and could not verify.

Phase 1c transforms abstention into a productive epistemic act.

When the controller abstains, this module:
    1. Identifies which signals drove the decision
    2. Constructs a structured uncertainty annotation
    3. Produces a Neti Neti reconstruction — verified residue

Neti Neti reconstruction:
    Named after the apophatic method in Advaita Vedanta —
    "not this, not this" — approaching what can be known by
    systematically eliminating what cannot be asserted.

    The system cannot say what is true.
    But it can say what is not verified, not grounded, not certain.
    What remains after systematic elimination is the verified residue —
    partial but every part is grounded.

    The system always returns something. What it returns, it stands behind.

Paper correspondence:
    Neti Neti — systematic elimination producing verified residue.
    The system never returns pure silence.

Output format:
    AbstentionResult contains:
        reason          : primary driver of abstention (signal name)
        signal_breakdown: dict of signal values that contributed
        annotation      : human-readable explanation of what drove abstention
        verified_residue: what can be stated with lower risk
        eliminated      : what was excluded and why
        confidence_note : honest characterisation of output reliability
        action_taken    : "abstain" | "partial"
"""

from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# SIGNAL THRESHOLDS FOR ANNOTATION
# These are used to categorise signal severity for the annotation,
# not for the controller decision (which uses distortion thresholds).
# =============================================================================

# Distortion thresholds for annotation severity
_HIGH_DISTORTION    = 0.42
_MEDIUM_DISTORTION  = 0.32

# State thresholds for annotation
_LOW_STABILITY      = 0.60
_HIGH_REACTIVITY    = 0.50
_HIGH_UNCERTAINTY   = 0.40
_LOW_INTEGRATION    = 0.45

# Trajectory thresholds for annotation
_HIGH_DRIFT         = 0.30
_HIGH_PEAK          = 0.45


# =============================================================================
# ABSTENTION RESULT
# =============================================================================

@dataclass
class AbstentionResult:
    """
    Structured result from the abstention module.

    Attributes:
        action_taken        : "abstain" or "partial"
        reason              : primary signal that drove abstention
        signal_breakdown    : dict of relevant signal values
        annotation          : human-readable explanation
        verified_residue    : what the system can state with lower risk
        eliminated          : list of (claim_type, reason) tuples
        confidence_note     : honest characterisation of reliability
        prompt_type         : the prompt taxonomy type
        D_combined          : distortion value that triggered abstention
    """
    action_taken:     str   = "abstain"
    reason:           str   = "elevated_distortion"
    signal_breakdown: dict  = field(default_factory=dict)
    annotation:       str   = ""
    verified_residue: str   = ""
    eliminated:       list  = field(default_factory=list)
    confidence_note:  str   = ""
    prompt_type:      str   = ""
    D_combined:       float = 0.0

    def to_dict(self) -> dict:
        return {
            "action_taken":     self.action_taken,
            "reason":           self.reason,
            "signal_breakdown": self.signal_breakdown,
            "annotation":       self.annotation,
            "verified_residue": self.verified_residue,
            "eliminated":       self.eliminated,
            "confidence_note":  self.confidence_note,
            "prompt_type":      self.prompt_type,
            "D_combined":       round(self.D_combined, 4),
        }

    def format_output(self) -> str:
        """
        Format the abstention result as a user-facing response.
        This is what the pipeline returns instead of silence.
        """
        lines = []

        lines.append("[UNCERTAINTY ANNOTATION]")
        lines.append(self.annotation)
        lines.append("")

        if self.verified_residue:
            lines.append("[VERIFIED RESIDUE]")
            lines.append(self.verified_residue)
            lines.append("")

        if self.eliminated:
            lines.append("[ELIMINATED — INSUFFICIENT GROUNDING]")
            for item, reason in self.eliminated:
                lines.append(f"  • {item}: {reason}")
            lines.append("")

        lines.append("[CONFIDENCE NOTE]")
        lines.append(self.confidence_note)

        return "\n".join(lines)


# =============================================================================
# SIGNAL ANALYSIS
# Determine which signals drove the abstention decision.
# =============================================================================

def _identify_primary_reason(
    state: dict,
    distortion: dict,
    pre_gen: Optional[dict],
    trajectory: Optional[object],
) -> tuple[str, dict]:
    """
    Identify the primary reason for abstention and relevant signal values.

    Returns (reason_string, signal_breakdown_dict).

    Priority order for reason identification:
        1. Pre-generation entity fabrication risk (strongest signal)
        2. High D_factual with low stability (factual grounding failure)
        3. High D_reasoning with low step consistency (reasoning breakdown)
        4. High trajectory drift (intra-output instability)
        5. High overall D_combined (general epistemic instability)
    """
    D_combined  = distortion["D_combined"]
    D_factual   = distortion["D_factual"]
    D_reasoning = distortion["D_reasoning"]
    S           = state["S"]
    R           = state["R"]
    I           = state["I"]
    SC          = state.get("step_consistency", 0.7)
    CS          = state.get("contradiction_score", 0.1)

    breakdown = {
        "D_combined":          round(D_combined, 4),
        "D_factual":           round(D_factual, 4),
        "D_reasoning":         round(D_reasoning, 4),
        "stability":           round(S, 4),
        "reactivity":          round(R, 4),
        "integration":         round(I, 4),
        "step_consistency":    round(SC, 4),
        "contradiction_score": round(CS, 4),
    }

    # 1. Pre-generation entity fabrication risk
    if pre_gen:
        screener = pre_gen.get("screener_result", {})
        agg_risk = screener.get("aggregate_risk", 0.0)
        ep       = pre_gen.get("epistemic_profile", {})
        if agg_risk > 0.6 or ep.get("fabrication_prone", False):
            breakdown["entity_fabrication_risk"] = round(agg_risk, 4)
            breakdown["fabrication_prone"]        = ep.get("fabrication_prone", False)
            high_risk = screener.get("high_risk_entities", [])
            if high_risk:
                breakdown["high_risk_entities"] = high_risk
            return "entity_fabrication_risk", breakdown

    # 2. High factual distortion with low stability
    if D_factual > _HIGH_DISTORTION and S < _LOW_STABILITY:
        return "factual_grounding_failure", breakdown

    # 3. High reasoning distortion with step consistency breakdown
    if D_reasoning > _HIGH_DISTORTION and SC < 0.5:
        return "reasoning_consistency_failure", breakdown

    # 4. Trajectory drift
    if trajectory and not trajectory.skipped:
        if trajectory.drift_score > _HIGH_DRIFT:
            breakdown["trajectory_drift"]   = round(trajectory.drift_score, 4)
            breakdown["trajectory_shape"]   = trajectory.trajectory_shape
            breakdown["peak_distortion"]    = round(trajectory.peak_distortion, 4)
            breakdown["peak_segment"]       = trajectory.peak_segment
            return "intra_output_instability", breakdown

    # 5. General elevated distortion
    return "elevated_distortion", breakdown


# =============================================================================
# ANNOTATION GENERATION
# Produce human-readable explanation of what drove the abstention.
# =============================================================================

_ANNOTATIONS = {
    "entity_fabrication_risk": (
        "The system identified named entities in this prompt that could not be "
        "verified in authoritative databases prior to generation. The output "
        "carries elevated risk of entity fabrication — the model may have "
        "generated plausible but unverifiable claims about entities whose "
        "existence has not been confirmed."
    ),
    "factual_grounding_failure": (
        "The output showed low semantic alignment with the prompt combined with "
        "elevated factual distortion. The generated content may not be reliably "
        "grounded in the prompt's factual requirements. The system cannot "
        "confirm the output's claims against available knowledge."
    ),
    "reasoning_consistency_failure": (
        "The output showed internal logical inconsistency. Reasoning steps were "
        "not fully coherent with each other, or the output contained "
        "contradictory statements. The system cannot confirm the logical "
        "validity of the conclusions drawn."
    ),
    "intra_output_instability": (
        "The output showed increasing epistemic instability across its sections. "
        "Early content was more grounded than later content — the generation "
        "drifted toward higher distortion as it progressed. The later sections "
        "of the output are less reliable than the opening."
    ),
    "elevated_distortion": (
        "The output showed elevated overall epistemic instability across "
        "multiple signal dimensions. The system cannot confirm the reliability "
        "of the output's claims with sufficient confidence to present them "
        "without qualification."
    ),
}

_CONFIDENCE_NOTES = {
    "entity_fabrication_risk": (
        "Entity existence could not be confirmed prior to generation. "
        "Claims about specific named entities should be independently verified "
        "before relying on this output."
    ),
    "factual_grounding_failure": (
        "Factual claims in this output should be independently verified. "
        "The system's distortion signals indicate the output may not be "
        "reliably grounded."
    ),
    "reasoning_consistency_failure": (
        "The logical structure of this output is uncertain. "
        "Conclusions should be independently verified before relying on them."
    ),
    "intra_output_instability": (
        "Earlier sections of this output are more reliable than later sections. "
        "Claims appearing later in the response carry higher uncertainty."
    ),
    "elevated_distortion": (
        "This output carries elevated epistemic uncertainty across multiple "
        "signal dimensions. Independent verification is recommended before "
        "relying on specific claims."
    ),
}


# =============================================================================
# NETI NETI RECONSTRUCTION
# Produce a verified residue from the output.
# =============================================================================

def _extract_verified_residue(
    output: str,
    reason: str,
    state: dict,
    trajectory: Optional[object],
) -> tuple[str, list]:
    """
    Extract what can be stated with lower risk — the verified residue.

    Strategy depends on the abstention reason:
        entity_fabrication_risk  → extract hedged general claims, exclude specific entity claims
        reasoning_consistency    → extract steps that were internally consistent
        intra_output_instability → extract early segments (lower distortion)
        factual/elevated         → extract hedged statements, exclude confident specifics

    Returns (verified_residue_str, eliminated_list).
    eliminated_list contains (claim_description, elimination_reason) tuples.
    """
    if not output or not output.strip():
        return "", []

    eliminated = []
    sentences  = [s.strip() for s in output.replace("\n", " ").split(".") if s.strip()]

    if not sentences:
        return "", []

    # For intra-output instability: use only early segments
    if reason == "intra_output_instability" and trajectory and not trajectory.skipped:
        peak = trajectory.peak_segment
        if peak > 0:
            # Return segments before the peak as verified residue
            early_segs = [
                s["segment_index"]
                for s in trajectory.segments
                if s["segment_index"] < peak and s["D_combined"] < 0.25
            ]
            if early_segs:
                eliminated.append((
                    f"Segments {peak} onward",
                    f"Distortion peaked at segment {peak} "
                    f"(D_combined={trajectory.peak_distortion:.3f}). "
                    f"Later content is less reliable."
                ))
                # Return just the opening statement as residue
                opening = sentences[0] if sentences else ""
                if opening:
                    return (
                        f"{opening}. "
                        f"[Note: only the opening statement is included. "
                        f"Later content showed elevated distortion.]",
                        eliminated
                    )

    # For entity fabrication: exclude confident specific claims
    if reason == "entity_fabrication_risk":
        eliminated.append((
            "Specific entity claims",
            "Named entities in the prompt could not be verified in "
            "authoritative databases. Claims about specific named entities "
            "are excluded from the verified residue."
        ))
        # Return only general/hedged statements
        hedged = [
            s for s in sentences
            if any(h in s.lower() for h in [
                "generally", "typically", "often", "may", "might",
                "in general", "as a rule", "broadly speaking",
                "uncertain", "unclear", "unknown", "not confirmed"
            ])
        ]
        if hedged:
            return ". ".join(hedged[:3]) + ".", eliminated
        else:
            return (
                "No verified residue could be extracted. "
                "All specific claims in this output require independent verification.",
                eliminated
            )

    # For reasoning consistency: return partial output with caveat
    if reason == "reasoning_consistency_failure":
        eliminated.append((
            "Logical conclusions",
            "Internal consistency of reasoning steps could not be confirmed. "
            "Conclusions derived from the reasoning chain are excluded."
        ))
        # Return setup/premises only, not conclusions
        if len(sentences) > 2:
            premises = sentences[:max(1, len(sentences) // 2)]
            return (
                ". ".join(premises) + ". [Conclusions excluded due to reasoning inconsistency.]",
                eliminated
            )

    # Default: return first two sentences as a minimal verified residue
    # with a caveat that the full output is uncertain
    eliminated.append((
        "Full output",
        f"Distortion D_combined={state['I']:.3f} exceeded abstention threshold. "
        f"Only opening statements included as partial residue."
    ))

    residue = ". ".join(sentences[:2]) + "."
    return (
        residue + " [Partial — full output uncertainty exceeds acceptance threshold.]",
        eliminated
    )


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_abstention_result(
    output: str,
    state: dict,
    distortion: dict,
    prompt_type: str = "",
    pre_gen: Optional[dict] = None,
    trajectory=None,
) -> AbstentionResult:
    """
    Build a structured abstention result.

    Called by the pipeline when the controller decision is "abstain".

    Args:
        output      : the generated output (before abstention)
        state       : cognitive state dict from state.py
        distortion  : distortion dict from distortion.py
        prompt_type : pipeline taxonomy type
        pre_gen     : pre-generation layer results dict
        trajectory  : TrajectoryResult from segment_tracker.py (or None)

    Returns:
        AbstentionResult with annotation, verified residue, and elimination log.
    """
    result = AbstentionResult()
    result.prompt_type = prompt_type
    result.D_combined  = distortion["D_combined"]

    # Identify primary reason and signal breakdown
    reason, breakdown = _identify_primary_reason(
        state, distortion, pre_gen, trajectory
    )
    result.reason           = reason
    result.signal_breakdown = breakdown

    # Generate annotation
    result.annotation     = _ANNOTATIONS.get(reason, _ANNOTATIONS["elevated_distortion"])
    result.confidence_note = _CONFIDENCE_NOTES.get(reason, _CONFIDENCE_NOTES["elevated_distortion"])

    # Neti Neti reconstruction
    residue, eliminated = _extract_verified_residue(
        output, reason, state, trajectory
    )
    result.verified_residue = residue
    result.eliminated       = eliminated

    # Determine action taken
    result.action_taken = "partial" if residue else "abstain"

    return result


def run_abstention(
    output: str,
    state: dict,
    distortion: dict,
    prompt_type: str = "",
    pre_gen: Optional[dict] = None,
    trajectory=None,
) -> AbstentionResult:
    """Public interface. Never raises."""
    try:
        return build_abstention_result(
            output, state, distortion,
            prompt_type=prompt_type,
            pre_gen=pre_gen,
            trajectory=trajectory,
        )
    except Exception:
        r = AbstentionResult()
        r.annotation      = "The system encountered an error during abstention processing."
        r.confidence_note = "This output could not be evaluated. Independent verification required."
        return r
