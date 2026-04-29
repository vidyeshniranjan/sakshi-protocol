import sys
import os
import json

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model import openai_model
from core.pipeline import SakshiPipeline

print("Starting benchmark run...")

# =========================
# CONFIGURATION
# =========================

MODE = "baseline"
# Options:
# "baseline"      — raw model output, no Sakshi observer
# "sakshi"        — Sakshi observer + distortion control, Omega disabled
# "sakshi_omega"  — Sakshi observer + distortion control + Omega grounding

PROMPTS_FILE = "prompts_test.json"

# =========================
# INITIALIZE
# =========================

if MODE == "baseline":
    pipeline = SakshiPipeline(openai_model, omega_enabled=False)
elif MODE == "sakshi":
    pipeline = SakshiPipeline(openai_model, omega_enabled=False)
elif MODE == "sakshi_omega":
    pipeline = SakshiPipeline(openai_model, omega_enabled=True)
else:
    raise ValueError(f"Invalid MODE: {MODE}")

# Load prompts
with open(PROMPTS_FILE) as f:
    prompts = json.load(f)

print(f"Loaded {len(prompts)} prompts | MODE={MODE}")

results = []

# =========================
# HELPER: evaluation
# =========================

def evaluate(output, gt):
    if not gt or not gt.strip():
        return None
    return int(gt.lower() in output.lower())

# =========================
# MAIN LOOP
# =========================

for item in prompts:
    print(f"Running prompt {item['id']}...")

    prompt = item["prompt"]
    gt = item.get("answer", "")

    try:
        if MODE == "baseline":
            output = pipeline.generator.generate(prompt)
            state = None
            distortion = None
            decision = "accept"
            intervened = False
            grounded = False

        else:
            output, state, distortion, decision, intervened, grounded = pipeline.run(prompt)

        correct = evaluate(output, gt)

        results.append({
            "id": item["id"],
            "type": item.get("type", ""),
            "prompt": prompt,
            "output": output,
            "ground_truth": gt,
            "correct": correct,
            "state": state,
            "distortion": distortion,
            "decision": decision,
            "intervened": intervened,
            "grounded": grounded
        })

    except Exception as e:
        print(f"Error on prompt {item['id']}: {e}")

# =========================
# SAVE RESULTS
# =========================

os.makedirs("../results", exist_ok=True)

output_file = f"../results/{MODE}.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
print("Benchmark run complete.")
