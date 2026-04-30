# Type-aware distortion thresholds.
#
# Reasoning prompts have a naturally higher distortion baseline because
# their outputs are longer, more variable in token distribution, and higher
# in reactivity — structural properties of multi-step generation, not
# indicators of unreliability. A single global threshold unfairly abstains
# on correct reasoning outputs.
#
# Factual / hallucination: standard thresholds.
# Reasoning / ambiguous: raised thresholds to account for natural variance.

THRESHOLDS = {
    "factual":       {"accept": 0.25, "retrieve": 0.35},
    "hallucination": {"accept": 0.25, "retrieve": 0.35},
    "reasoning":     {"accept": 0.32, "retrieve": 0.42},
    "ambiguous":     {"accept": 0.30, "retrieve": 0.40},
}

DEFAULT_THRESHOLDS = {"accept": 0.25, "retrieve": 0.35}


def decide(state, distortion, prompt_type=""):
    t = THRESHOLDS.get(prompt_type, DEFAULT_THRESHOLDS)

    if distortion < t["accept"]:
        return "accept"
    elif distortion < t["retrieve"]:
        return "retrieve"
    else:
        return "abstain"
