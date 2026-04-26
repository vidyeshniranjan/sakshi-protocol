from openai import OpenAI

client = OpenAI()

def openai_model(prompt):
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content
