def decide(distortion, threshold=0.6):
    if distortion < 0.3:
        return "accept"
    elif distortion < threshold:
        return "revise"
    else:
        return "abstain"
