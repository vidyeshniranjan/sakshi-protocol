def compute_state(signals: dict) -> dict:
    """
    Map extracted signals to the cognitive state vector Ct = [S, R, T, V, I].

    V3 changes vs V2:
    - step_consistency and contradiction_score consumed from signals
      for reasoning prompts (neutral defaults for all other types).
    - Integration (I) incorporates step_consistency and contradiction
      as additional dimensions of reasoning coherence.

    V3.1 calibration fix:
    - T and V formulations restored to V2/V3 original (specificity-based).
    - hallucination_risk restored to conf * spec * (1 - unc).
    - semantic_deviation fix abandoned: semantic similarity cannot
      distinguish confident correct answers from confident fabrications,
      making it unsuitable as a hallucination risk proxy. This is
      consistent with the Class 2 boundary established in V2.

    State variables are all in [0, 1].
    Higher S and I = more stable/coherent = lower distortion.
    Higher R, T, V = more reactive/transformed/biased = higher distortion.
    """

    sim           = signals["similarity"]
    coh           = signals["coherence"]
    length        = signals["length_score"]
    unc           = signals["uncertainty"]
    spec          = signals["specificity"]
    conf          = signals["confidence"]
    step_cons     = signals["step_consistency"]
    contradiction = signals["contradiction_score"]

    # S — Stability
    S = min(sim * 1.1, 1.0)

    # R — Reactivity
    R = 1 - coh

    # T — Transformation (specificity-based, softened)
    T = spec * 0.3

    # V — Valuation (length bias)
    V = abs(0.5 - length) * 0.2

    # hallucination_risk: confident + specific + unhedged
    hallucination_risk = conf * spec * (1 - unc)

    # I — Integration
    # V3: incorporates step_consistency and contradiction for reasoning prompts
    I_base = coh * (1 - unc) * (1 - 0.3 * conf * spec)
    I = (
        I_base
        * (1 - 0.1 * hallucination_risk)
        * (0.8 + 0.2 * step_cons)
        * (1 - 0.15 * contradiction)
    )
    I = max(0.0, min(I, 1.0))

    return {
        "S":  round(S, 6),
        "R":  round(R, 6),
        "T":  round(T, 6),
        "V":  round(V, 6),
        "I":  round(I, 6),
        "step_consistency":    round(step_cons, 6),
        "contradiction_score": round(contradiction, 6),
    }
