from openai import OpenAI
import numpy as np

client = OpenAI()


# --- Embedding ---
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- Uncertainty ---
def uncertainty_score(output):
    hedges = [
        "i don't know",
        "i am not aware",
        "as of my last knowledge",
        "it is possible",
        "may not be",
        "not well documented",
        "unclear",
        "no evidence"
    ]

    output_lower = output.lower()
    score = sum(1 for h in hedges if h in output_lower)

    return min(score / 3, 1.0)


# --- Main ---
def extract_signals(prompt, output):

    emb_prompt = get_embedding(prompt)
    emb_output = get_embedding(output)

    similarity = cosine_similarity(emb_prompt, emb_output)

    length = len(output.split())
    length_score = min(length / 100, 1.0)

    coherence = len(set(output.split())) / max(len(output.split()), 1)

    uncertainty = uncertainty_score(output)

    return {
        "similarity": float(similarity),
        "length_score": float(length_score),
        "coherence": float(coherence),
        "uncertainty": float(uncertainty)
    }
def specificity_score(output):
    words = output.split()

    # longer + structured answers = higher risk
    return min(len(words) / 150, 1.0)
