import pandas as pd
import numpy as np
import scipy.stats as st

# Load results
df = pd.read_json("../results/sakshi_omega.json")  # change per run

# --- BASIC METRICS ---

# Only consider rows where correctness exists
valid = df[df["correct"].notnull()]

accuracy = valid["correct"].mean()
hallucination = 1 - accuracy

retrieval_rate = (df["decision"] == "retrieve").mean()
abstention_rate = (df["decision"] == "abstain").mean()

avg_distortion = df["distortion"].dropna().mean()

print("=== METRICS ===")
print(f"Accuracy: {accuracy:.3f}")
print(f"Hallucination Rate: {hallucination:.3f}")
print(f"Retrieval Rate: {retrieval_rate:.3f}")
print(f"Abstention Rate: {abstention_rate:.3f}")
print(f"Avg Distortion: {avg_distortion:.3f}")

# --- CONFIDENCE INTERVAL ---

def confidence_interval(data):
    if len(data) < 2:
        return (None, None)
    mean = np.mean(data)
    ci = st.t.interval(0.95, len(data)-1, loc=mean, scale=st.sem(data))
    return ci

ci_low, ci_high = confidence_interval(valid["correct"])

print("\n=== CONFIDENCE INTERVAL ===")
print(f"Accuracy CI: ({ci_low:.3f}, {ci_high:.3f})")

# --- DECISION DISTRIBUTION ---

print("\n=== DECISION COUNTS ===")
print(df["decision"].value_counts())

# --- DISTORTION ANALYSIS ---

print("\n=== DISTORTION ANALYSIS ===")
print(df[["distortion", "correct", "decision"]].head(10))
