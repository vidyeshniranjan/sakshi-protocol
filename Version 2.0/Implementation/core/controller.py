def decide(state, distortion):

    if distortion < 0.20:
        return "accept"

    elif distortion < 0.30:
        return "retrieve"   # Ω activation zone

    else:
        return "abstain"
