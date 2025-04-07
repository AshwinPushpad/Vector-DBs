from fastapi import FastAPI
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize FastAPI app
app = FastAPI()

@app.post("/")
async def home():
    return 'hello'

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="Text Similarity Collection")
collection = Collection("text_similarity", schema)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# API to insert text
def insert_text(texts):
    embeddings = [model.encode(text).tolist() for text in texts]
    data = [[text for text in texts], embeddings]
    collection.insert(data)
    collection.flush()

@app.post("/insert/")
async def insert_api(texts: list[str]):
    insert_text(texts)
    return {"message": "Texts inserted successfully"}

# API to search for similar text
@app.get("/search/")
async def search_api(query: str, top_k: int = 5):
    query_embedding = model.encode(query).tolist()
    collection.load()
    results = collection.search([query_embedding], "embedding", search_params={"metric_type": "COSINE"}, limit=top_k)
    return {"matches": results}

# Run using: uvicorn main:app --reload