def decide(state, distortion):
    if distortion < 0.3:
        return "accept"
    elif distortion < 0.4:
        return "retrieve"   
    elif distortion < 0.6:
        return "revise"
    else:
        return "abstain"
