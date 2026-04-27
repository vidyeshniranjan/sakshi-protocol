from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_signals(prompt, output):

    emb_prompt = get_embedding(prompt)
    emb_output = get_embedding(output)

    similarity = cosine_similarity(emb_prompt, emb_output)

    # Keep simple proxies for now
    length = len(output.split())
    length_score = min(length / 100, 1.0)

    coherence = len(set(output.split())) / max(len(output.split()), 1)

    return {
        "similarity": float(similarity),
        "length_score": float(length_score),
        "coherence": float(coherence)
    }
