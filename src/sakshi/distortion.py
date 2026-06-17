"""
distortion.py — Sakshi-Protocol V3

Split distortion formulation.

V3 vs V2:
    V2 used a single scalar distortion over all state variables uniformly,
    conflating structural output properties with epistemic instability.

    V3 separates into two dimensions:
        D_factual   — epistemic instability in knowledge/recall outputs
        D_reasoning — logical instability in multi-step generation
        D_combined  — prompt-type-weighted blend, primary controller input

V3.1 calibration note:
    _wH restored to 2.5 (original V3 value). The semantic_deviation
    approach to H was abandoned — see state.py for rationale.
    H approximation from T and V retained as original.
"""

# Factual distortion weights
_wS  = 1.5
_wV  = 0.3
_wH  = 2.5
_FACTUAL_WEIGHT_SUM = _wS + _wV + _wH   # = 4.3

# Reasoning distortion weights
_wR  = 1.0
_wT  = 0.8
_wI  = 2.0
_wSC = 1.5
_wCS = 1.2
_REASONING_WEIGHT_SUM = _wR + _wT + _wI + _wSC + _wCS  # = 6.5

# Mixing weights per prompt type
_MIXING_WEIGHTS = {
    "factual":       {"factual": 0.85, "reasoning": 0.15},
    "hallucination": {"factual": 0.90, "reasoning": 0.10},
    "reasoning":     {"factual": 0.20, "reasoning": 0.80},
    "ambiguous":     {"factual": 0.50, "reasoning": 0.50},
}
_DEFAULT_MIXING = {"factual": 0.60, "reasoning": 0.40}


def _hallucination_risk(state: dict) -> float:
    S = state["S"]
    T = state["T"]
    V = state["V"]
    I = state["I"]
    approx_confidence  = min((V / 0.2) * 0.5 + (T / 0.3) * 0.5, 1.0)
    approx_specificity = min(T / 0.3, 1.0)
    approx_uncertainty = max(0.0, (1 - I) * 0.5 + (1 - S) * 0.5)
    risk = approx_confidence * approx_specificity * (1 - approx_uncertainty)
    return max(0.0, min(risk, 1.0))


def compute_distortion(state: dict, prompt_type: str = "") -> dict:
    """
    Compute split distortion over the cognitive state vector.

    Returns dict with D_factual, D_reasoning, D_combined.
    D_combined is the primary controller input.
    """

    S  = state["S"]
    R  = state["R"]
    T  = state["T"]
    V  = state["V"]
    I  = state["I"]
    SC = state.get("step_consistency", 0.7)
    CS = state.get("contradiction_score", 0.1)

    H = _hallucination_risk(state)

    # D_factual
    D_factual_raw = (
        _wS * (1 - S) +
        _wV * V +
        _wH * H
    )
    D_factual = max(0.0, min(D_factual_raw / _FACTUAL_WEIGHT_SUM, 1.0))

    # D_reasoning
    D_reasoning_raw = (
        _wR * R +
        _wT * T +
        _wI * (1 - I) +
        _wSC * (1 - SC) +
        _wCS * CS
    )
    D_reasoning = max(0.0, min(D_reasoning_raw / _REASONING_WEIGHT_SUM, 1.0))

    # D_combined
    mix = _MIXING_WEIGHTS.get(prompt_type, _DEFAULT_MIXING)
    D_combined = (
        mix["factual"]   * D_factual +
        mix["reasoning"] * D_reasoning
    )
    D_combined = max(0.0, min(D_combined, 1.0))

    return {
        "D_factual":   round(D_factual,   6),
        "D_reasoning": round(D_reasoning, 6),
        "D_combined":  round(D_combined,  6),
    }
