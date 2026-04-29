def decide(state, distortion):

    if distortion < 0.22:
        return "accept"

    elif distortion < 0.32:
        return "retrieve"   # Ω activation zone

    else:
        return "abstain"
