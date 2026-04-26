def decide(state, distortion):
    if distortion < 0.3:
        return "accept"
    elif distortion < 0.6:
        return "revise"
    else:
        # high distortion → decide if retrieval or abstain
        if state["I"] < 0.4:   # low integration = grounding issue
            return "retrieve"
        else:
            return "abstain"
