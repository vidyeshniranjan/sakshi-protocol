def compute_state(signals):
    sim = signals["similarity"]
    length = signals["length_score"]
    coh = signals["coherence"]

    S = sim
    R = 1 - sim
    T = 1 - sim
    V = 1 - length
    I = coh

    return {
        "S": S,
        "R": R,
        "T": T,
        "V": V,
        "I": I
    }
