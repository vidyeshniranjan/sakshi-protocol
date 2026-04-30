from core.generator import Generator
from core.signals import extract_signals
from core.state import compute_state
from core.distortion import compute_distortion
from core.controller import decide
from core.retriever import retrieve


# Prompt types where Omega grounding is meaningful.
# Reasoning prompts excluded: no external document can help solve a logic
# puzzle or math problem, and retrieved context actively misleads
# regeneration by introducing irrelevant factual framing.
GROUNDABLE_TYPES = {"factual", "hallucination"}

# Post-grounding fallback abstain threshold
POST_GROUNDING_ABSTAIN_THRESHOLD = 0.35


class SakshiPipeline:
    def __init__(self, model_fn, omega_enabled=False):
        self.generator = Generator(model_fn)
        self.omega_enabled = omega_enabled

    def run(self, prompt, prompt_type=""):
        # intervened: True only when the controller produced a non-accept
        # decision — i.e., when the system actually changed behaviour.
        # accept = observer ran passively, no intervention occurred.
        #
        # grounded: True only when Omega was invoked and regeneration occurred.
        intervened = False
        grounded = False
        distortion_pre_grounding = None

        # Step 1: generate
        output = self.generator.generate(prompt)

        # Step 2: extract signals
        signals = extract_signals(prompt, output)

        # Step 3: compute state
        state = compute_state(signals)

        # Step 4: compute distortion
        distortion = compute_distortion(state)

        # Step 5: type-aware decision
        decision = decide(state, distortion, prompt_type=prompt_type)

        # Step 6: mark intervention if controller fired
        if decision in ("retrieve", "abstain"):
            intervened = True

        # Step 7: Omega grounding
        if decision == "retrieve" and self.omega_enabled:

            if prompt_type in GROUNDABLE_TYPES:
                # Record pre-grounding distortion — this is what triggered
                # the retrieve decision and is the value to report in the paper.
                distortion_pre_grounding = distortion
                grounded = True

                context = retrieve(prompt)

                grounded_prompt = f"""Answer the question using ONLY the verified information below.

Question:
{prompt}

Verified Context:
{context}

If the information is uncertain, say so clearly."""

                output = self.generator.generate(grounded_prompt)

                # Recompute after grounding
                signals = extract_signals(prompt, output)
                state = compute_state(signals)
                distortion = compute_distortion(state)
                decision = decide(state, distortion, prompt_type=prompt_type)

                # Fallback abstain if still unstable after grounding
                if distortion > POST_GROUNDING_ABSTAIN_THRESHOLD and state["I"] < 0.4:
                    decision = "abstain"

            else:
                # retrieve on non-groundable type: abstain
                # (no external source can help with reasoning/ambiguous prompts)
                decision = "abstain"

        elif decision == "retrieve" and not self.omega_enabled:
            # Sakshi-only mode: retrieve signals risk but Omega unavailable
            decision = "abstain"

        return output, state, distortion, distortion_pre_grounding, decision, intervened, grounded
