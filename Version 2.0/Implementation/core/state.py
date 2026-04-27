def compute_state(signals):
    sim = signals["similarity"]
    coh = signals["coherence"]
    length = signals["length_score"]
    unc = signals["uncertainty"]
    spec = signals["specificity"]
    conf = signals["confidence"]

    # Stability
    S = min(sim * 1.1, 1.0)

    # Reactivity
    R = 1 - coh

    # Transformation (softened)
    T = spec * 0.3

    # Variability (very weak now)
    V = abs(0.5 - length) * 0.2

    # 🔥 Hallucination risk
    hallucination_risk = conf * spec * (1 - unc)

    # Integration (base)
    I = coh * (1 - unc) * (1 - 0.3 * conf * spec)

    # 🔥 Final adjustment (key fix)
    I = I * (1 - 0.5 * hallucination_risk)

    return {
        "S": S,
        "R": R,
        "T": T,
        "V": V,
        "I": I
    }
