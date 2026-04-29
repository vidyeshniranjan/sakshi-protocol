from core.generator import Generator
from core.signals import extract_signals
from core.state import compute_state
from core.distortion import compute_distortion
from core.controller import decide
from core.retriever import retrieve


class SakshiPipeline:
    def __init__(self, model_fn, omega_enabled=False):
        self.generator = Generator(model_fn)
        self.omega_enabled = omega_enabled

    def run(self, prompt):
        # intervened: True whenever the Sakshi observer computed state and
        # produced a non-trivial control signal (i.e., always in Sakshi mode).
        # This reflects the paper's notion of state-aware monitoring being active.
        # Separate from grounded, which records whether Omega was actually invoked.
        intervened = True
        grounded = False

        # Step 1: initial generation
        output = self.generator.generate(prompt)

        # Step 2: extract signals
        signals = extract_signals(prompt, output)

        # Step 3: compute state
        state = compute_state(signals)

        # Step 4: compute distortion
        distortion = compute_distortion(state)

        # Step 5: decision
        decision = decide(state, distortion)

        # Step 6: Omega retrieval (grounding) — only if omega_enabled
        if decision == "retrieve" and self.omega_enabled:
            grounded = True

            context = retrieve(prompt)

            grounded_prompt = f"""
Answer the question using ONLY the verified information below.

Question:
{prompt}

Verified Context:
{context}

If the information is uncertain, say so clearly.
"""

            # regenerate grounded answer
            output = self.generator.generate(grounded_prompt)

            # recompute signals after grounding
            signals = extract_signals(prompt, output)
            state = compute_state(signals)
            distortion = compute_distortion(state)
            decision = decide(state, distortion)

            # fallback abstain if still unstable after grounding
            if distortion > 0.35 and state["I"] < 0.4:
                decision = "abstain"

        elif decision == "retrieve" and not self.omega_enabled:
            # In Sakshi-only mode (no Omega), retrieve signals elevated risk
            # but no grounding is available — treat as abstain to avoid
            # outputting a potentially unreliable response
            decision = "abstain"

        return output, state, distortion, decision, intervened, grounded
