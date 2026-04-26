import pandas as pd

df = pd.read_json("../experiments/results_test.json")

print("Avg Distortion:", df["distortion"].mean())
print("Decision counts:")
print(df["decision"].value_counts())
