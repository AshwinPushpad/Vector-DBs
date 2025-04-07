import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file
# Set your OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import openai
openai.api_key = OPENAI_API_KEY  # Replace with your key

class OpenAIEmbedding:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model

    def encode(self, text):
        response = openai.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding
    def decode(self, embedding):
        pass
        response = openai.embeddings.create(input=[embedding], model=self.model)
        return response.data[0].embedding

# Create an instance just like SentenceTransformer
model = OpenAIEmbedding()
# Load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast model

# Initialize ChromaDB (Persistent Storage)
client = chromadb.Client()
# client = chromadb.PersistentClient(path="./chroma_db")
# client.delete_collection('documents')
collection = client.get_or_create_collection("documents")



# Sample documents
documents = [
    {"id": "1", "text": "The Eiffel Tower is a famous landmark in italy."},
    {"id": "2", "text": "Albert Einstein proposed the theory of relativity."},
    {"id": "3", "text": "Python is widely used in AI and machine learning applications."},
    {"id": "4", "text": "I am ashwin sisodiya, and i lives in indore studying btech."},
    {"id": "5", "name ": "arpan",'city':'indore','age':22,"profession":"teacher at ips"},
    {"id": "6", "name ": "ashwin pratap",'city':'bhopal','age':24,"profession":"student at ips"},
    {"id": "7", "name ": "arpan",'city':'delhi','age':25,"profession":"painter at ips"},
    {"id": "8", "name ": "ashwini",'city':'noida','age':23,"profession":"cricketer at ips"},
    {"id": "9", "name ": "arpan",'city':'punjab','age':21,"profession":"musician at ips"},
    {"id": "10", "text": "The Eiffel Tower is a famous landmark in paris."},
    {"id": "20", "text": "The colloseum is a famous in italy."},

]

# Convert to embeddings and store in ChromaDB
for doc in documents:
    embedding = model.encode(str(doc))
    # print(embedding)
    collection.add(ids=[doc["id"]], embeddings=[embedding], metadatas=[doc])

def retrieve_relevant_docs(query, top_k=2):
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    print(results)
    return [res for res in results["metadatas"][0]]

# # Test Retrieval
# retrieved_docs = retrieve_relevant_docs("Where is the Eiffel Tower?")
# print("Retrieved Docs:", retrieved_docs)

from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file
# Set your OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_answer(query, retrieved_docs):
    # context = "\n".join(retrieved_docs)
    context =retrieved_docs
    print(context)
    prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


# Test the RAG pipeline
query = "whats age of ashwin?"
retrieved_docs = retrieve_relevant_docs(query)

# print(docs  for docs in retrieve_relevant_docs)
answer = generate_answer(query, retrieved_docs)

print("Generated Answer:", answer)
