import json
from core.pipeline import SakshiPipeline

def dummy_model(prompt):
    return f"Generated response for: {prompt}"

pipeline = SakshiPipeline(dummy_model)

with open("prompts_test.json") as f:
    prompts = json.load(f)

results = []

for item in prompts:
    output, state, distortion, decision = pipeline.run(item["prompt"])

    results.append({
        "id": item["id"],
        "type": item["type"],
        "prompt": item["prompt"],
        "output": output,
        "state": state,
        "distortion": distortion,
        "decision": decision
    })

with open("results_test.json", "w") as f:
    json.dump(results, f, indent=2)

print("Test run complete.")
