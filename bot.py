from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

# Send structured data to OpenAI for answer generation
def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    print(context)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"What is Ashwin’s age? Here’s the data: {retrieved_text}"}
    ]
    )
    
    return response.choices[0].message.content

# Test the RAG pipeline
query = "where does ashwin live?"

# Convert vectors back to text (optional, if needed)
from multi_vector_milvus_db import retrieved_info
retrieved_text = model.decode(retrieved_info["age"])
retrieved_docs = retrieve_docs(query)

# print(docs  for docs in retrieve_relevant_docs)
answer = generate_answer(query, retrieved_docs)

print("Generated Answer:", answer)
