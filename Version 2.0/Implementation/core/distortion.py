def compute_distortion(state):
    w = {"S": 1, "R": 1, "T": 1, "V": 1, "I": 1}

    D = (
        w["S"] * (1 - state["S"]) +
        w["R"] * state["R"] +
        w["T"] * state["T"] +
        w["V"] * state["V"] +
        w["I"] * (1 - state["I"])
    )

    return max(0, min(D / 5, 1))
