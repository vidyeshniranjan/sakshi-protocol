import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model import openai_model
from core.pipeline import SakshiPipeline

print("Starting test run...")

# Initialize pipeline
pipeline = SakshiPipeline(openai_model)

# Load prompts
with open("prompts_test.json") as f:
    prompts = json.load(f)

print(f"Loaded {len(prompts)} prompts")

results = []

# Run experiment loop
for item in prompts:
    print(f"Running prompt {item['id']}...")

    try:
        output, state, distortion, decision, intervened = pipeline.run(prompt)

        print("Output:", output[:60])  # preview

        results.append({
            "id": item["id"],
            "type": item["type"],
            "prompt": item["prompt"],
            "output": output,
            "state": state,
            "distortion": distortion,
            "decision": decision,
            "intervened": intervened
        })

    except Exception as e:
        print(f"Error on prompt {item['id']}:", e)

# Save results
output_file = "results_test.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
print("Test run complete.")
