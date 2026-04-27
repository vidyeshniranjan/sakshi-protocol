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
        # Step 1: initial generation
        output = self.generator.generate(prompt)

        # Step 2: extract signals (FIXED)
        signals = extract_signals(prompt, output)

        # Step 3: compute state
        state = compute_state(signals)

        # Step 4: compute distortion
        distortion = compute_distortion(state)

        # Step 5: decision
        decision = decide(state, distortion)

        # Step 6: retrieval (optional intervention)
        if decision == "retrieve":
            context = retrieve(prompt)
            output = self.generator.generate(prompt + "\n\n" + context)

            # Recompute after intervention
            signals = extract_signals(prompt, output)
            state = compute_state(signals)
            distortion = compute_distortion(state)

        return output, state, distortion, decision
