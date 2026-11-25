from src.llm import client

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain what Retrieval-Augmented Generation is in two sentences."
)

print(response.text)