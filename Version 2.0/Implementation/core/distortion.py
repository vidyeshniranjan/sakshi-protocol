def compute_distortion(state):
   w = {
    "S": 1.5,   # reward alignment more
    "R": 0.7,
    "T": 0.5,   # reduce verbosity penalty
    "V": 0.3,   # reduce length penalty
    "I": 1.5    # reward integration more
}

    D = (
        w["S"] * (1 - state["S"]) +
        w["R"] * state["R"] +
        w["T"] * state["T"] +
        w["V"] * state["V"] +
        w["I"] * (1 - state["I"])
    )

    return max(0, min(D / 5, 1))
