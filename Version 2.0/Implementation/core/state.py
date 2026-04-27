def compute_state(signals):
    sim = signals["similarity"]
    coh = signals["coherence"]
    length = signals["length_score"]
    unc = signals["uncertainty"]

    S = sim
    R = 1 - sim
    T = 1 - sim
    V = 1 - length

    # Major change
    I = coh * (1 - unc)

    return {
        "S": S,
        "R": R,
        "T": T,
        "V": V,
        "I": I
    }
