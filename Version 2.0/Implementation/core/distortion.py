def compute_distortion(state):
    w = {
        "S": 1.5,
        "R": 0.6,
        "T": 0.4,
        "V": 0.2,
        "I": 2.0
    }

    S = state["S"]
    R = state["R"]
    T = state["T"]
    V = state["V"]
    I = state["I"]

    D = (
        w["S"] * (1 - S) +
        w["R"] * R +
        w["T"] * T +
        w["V"] * V +
        w["I"] * (1 - I)
    )

    return max(0, min(D / 5, 1))
