def decide(state, distortion):
    if distortion < 0.25:
        return "accept"
    elif distortion < 0.5:
        return "revise"
    elif distortion < 0.7:
        if state["I"] < 0.4:
            return "retrieve"
        return "abstain"
    else:
        return "abstain"
