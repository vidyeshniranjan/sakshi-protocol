def compute_state(signals):
    sim     = signals["similarity"]
    coh     = signals["coherence"]
    length  = signals["length_score"]
    unc     = signals["uncertainty"]
    spec    = signals["specificity"]
    conf    = signals["confidence"]

    # S — Stability: semantic alignment between prompt and output.
    S = min(sim * 1.1, 1.0)

    # R — Reactivity: inverse of lexical coherence.
    R = 1 - coh

    # T — Transformation: deviation from input context.
    # Softened — long outputs are not inherently unreliable.
    T = spec * 0.3

    # V — Valuation: length bias signal. Very weak contributor.
    V = abs(0.5 - length) * 0.2

    # hallucination_risk: confident + specific + not hedged.
    # This combination characterises the fluent-but-ungrounded failure mode.
    hallucination_risk = conf * spec * (1 - unc)

    # I — Integration: coherence penalised by hallucination risk.
    # Penalty weight reduced from 0.5 to 0.2 to stop over-penalising
    # reasoning outputs whose conf/spec are naturally higher due to
    # multi-step structure — not because they are fabricating.
    I_base = coh * (1 - unc) * (1 - 0.3 * conf * spec)
    I = I_base * (1 - 0.1 * hallucination_risk)

    return {
        "S": round(S, 6),
        "R": round(R, 6),
        "T": round(T, 6),
        "V": round(V, 6),
        "I": round(I, 6)
    }
