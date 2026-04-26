import numpy as np

def extract_signals(output):
    return {
        "entropy": np.random.uniform(0.2, 0.8),
        "similarity": np.random.uniform(0.3, 0.9),
        "variance": np.random.uniform(0.1, 0.7),
        "length": len(output.split())
    }
