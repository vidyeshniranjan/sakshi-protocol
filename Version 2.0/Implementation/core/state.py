def compute_state(signals):
    sim = signals["similarity"]
    coh = signals["coherence"]
    length = signals["length_score"]
    unc = signals["uncertainty"]
    spec = signals["specificity"]
    conf = signals["confidence"]

    # FIX 4 — slight boost to stability
    S = min(sim * 1.1, 1.0)

    # Decoupled dimensions
    R = 1 - coh

    # FIX 1 — soften specificity impact
    T = spec * 0.3

    # FIX 3 — balanced verbosity penalty
    V = abs(0.5 - length) * 0.2

    # FIX 2 — reduce confidence penalty
    I = coh * (1 - unc) * (1 - 0.3 * conf * spec)

    return {
        "S": S,
        "R": R,
        "T": T,
        "V": V,
        "I": I
    }
