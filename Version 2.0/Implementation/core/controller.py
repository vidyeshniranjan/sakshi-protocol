def decide(state, distortion):
    if distortion < 0.38:   # ← was 0.35
        return "accept"
    elif distortion < 0.55: # ← was 0.5
        return "revise"
    elif distortion < 0.65:
        if state["I"] < 0.4:
            return "retrieve"
        return "abstain"
    else:
        return "abstain"
