def compute_state(signals):
    S = 1 - signals["variance"]
    R = signals["entropy"]
    T = 1 - signals["similarity"]
    V = signals["entropy"] * 0.5
    I = signals["similarity"]

    return {
        "S": S,
        "R": R,
        "T": T,
        "V": V,
        "I": I
    }
