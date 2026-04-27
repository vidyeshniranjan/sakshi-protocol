from openai import OpenAI

client = OpenAI()


def retrieve(prompt):
    """
    Simple Ω implementation:
    Ask model to provide factual grounding / verification context.
    """

    query = f"""
You are a verification system.

Given the query below, provide:
1. Known factual information (if it exists)
2. Say clearly if the premise is uncertain or possibly incorrect

Query:
{prompt}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        temperature=0
    )

    return response.choices[0].message.content
