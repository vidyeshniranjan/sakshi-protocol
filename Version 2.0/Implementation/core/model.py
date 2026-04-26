from openai import OpenAI

client = OpenAI()

def openai_model(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print("API Error:", e)
        return "ERROR"
