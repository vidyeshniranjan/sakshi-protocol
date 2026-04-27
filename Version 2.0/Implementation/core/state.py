def compute_state(signals):
    sim = signals["similarity"]
    coh = signals["coherence"]
    length = signals["length_score"]
    unc = signals["uncertainty"]
    spec = signals["specificity"]
    conf = signals["confidence"]

    S = sim

    # decoupled now
    R = 1 - coh                # instability in structure
    T = spec                  # transformation / verbosity
    V = 1 - length            # brevity / confidence proxy

    # main upgrade
    I = coh * (1 - unc) * (1 - conf * spec)

    return {
        "S": S,
        "R": R,
        "T": T,
        "V": V,
        "I": I
    }
