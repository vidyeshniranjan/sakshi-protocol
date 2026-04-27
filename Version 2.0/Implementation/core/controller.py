def decide(state, distortion):

    if distortion < 0.25:
        return "accept"

    elif distortion < 0.4:
        return "retrieve"   # Ω activation zone

    else:
        return "abstain"
