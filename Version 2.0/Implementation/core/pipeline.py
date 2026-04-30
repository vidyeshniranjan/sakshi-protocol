from core.generator import Generator
from core.signals import extract_signals
from core.state import compute_state
from core.distortion import compute_distortion
from core.controller import decide
from core.retriever import retrieve


# Prompt types where external grounding is beneficial.
# Pure reasoning prompts (logic puzzles, math, multi-step inference) are
# excluded because external retrieval cannot help solve them and may
# actively mislead regeneration by introducing irrelevant context.
GROUNDABLE_TYPES = {"factual", "hallucination"}


class SakshiPipeline:
    def __init__(self, model_fn, omega_enabled=False):
        self.generator = Generator(model_fn)
        self.omega_enabled = omega_enabled

    def run(self, prompt, prompt_type=""):
        # intervened: True whenever the Sakshi observer computed state and
        # produced a non-trivial control signal (always True in Sakshi mode).
        # grounded: True only when Omega was actually invoked.
        intervened = True
        grounded = False
        distortion_pre_grounding = None

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

        # Step 6: Omega retrieval (grounding)
        if decision == "retrieve" and self.omega_enabled:

            if prompt_type in GROUNDABLE_TYPES:
                # Save pre-grounding distortion before regeneration overwrites it.
                # This allows the paper to show that grounded cases had genuinely
                # elevated distortion before Omega was invoked.
                distortion_pre_grounding = distortion
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

                # Regenerate grounded answer
                output = self.generator.generate(grounded_prompt)

                # Recompute signals and state after grounding
                signals = extract_signals(prompt, output)
                state = compute_state(signals)
                distortion = compute_distortion(state)
                decision = decide(state, distortion)

                # Fallback abstain if still unstable after grounding
                if distortion > 0.35 and state["I"] < 0.4:
                    decision = "abstain"

            else:
                # Reasoning/ambiguous prompts: retrieve signals elevated risk
                # but external grounding cannot help — abstain instead.
                decision = "abstain"

        elif decision == "retrieve" and not self.omega_enabled:
            # Sakshi-only mode: no Omega available — treat retrieve as abstain.
            decision = "abstain"

        return output, state, distortion, distortion_pre_grounding, decision, intervened, grounded
