from core.generator import Generator
from core.signals import extract_signals
from core.state import compute_state
from core.distortion import compute_distortion
from core.controller import decide
from core.retriever import retrieve


class SakshiPipeline:
    def __init__(self, model_fn):
        self.generator = Generator(model_fn)

    def run(self, prompt):

        intervened = False  # 🔥 NEW

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

        # Step 6: Ω retrieval (grounding)
        if decision == "retrieve":
            intervened = True  # 🔥 mark intervention

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

            #OPTIONAL: fallback abstain
            if distortion > 0.35 and state["I"] < 0.4:
                decision = "abstain"

        return output, state, distortion, decision, intervened
