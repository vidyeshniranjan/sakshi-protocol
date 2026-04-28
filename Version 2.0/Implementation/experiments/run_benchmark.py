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

MODE = "sakshi"  
# Options:
# "baseline"
# "sakshi"
# "sakshi_omega"

PROMPTS_FILE = "prompts_test.json"  # your existing prompts

# =========================
# INITIALIZE
# =========================

pipeline = SakshiPipeline(openai_model)

# Load prompts
with open(PROMPTS_FILE) as f:
    prompts = json.load(f)

print(f"Loaded {len(prompts)} prompts")

results = []

# =========================
# HELPER: evaluation
# =========================

def evaluate(output, gt):
    if not gt:
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

elif MODE == "sakshi":
    output, state, distortion, decision, intervened = pipeline.run(prompt)

elif MODE == "sakshi_omega":
    output, state, distortion, decision, intervened = pipeline.run(prompt)

else:
    raise ValueError(f"Invalid MODE: {MODE}")

        # -------------------------
        # EVALUATION
        # -------------------------
        correct = evaluate(output, gt)

        print("Output:", output[:60])  # preview

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
            "intervened": intervened
        })

    except Exception as e:
        print(f"Error on prompt {item['id']}:", e)

# =========================
# SAVE RESULTS
# =========================

# Ensure results folder exists
os.makedirs("../results", exist_ok=True)

output_file = f"../results/{MODE}.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
print("Benchmark run complete.")
