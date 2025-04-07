import chromadb

class ChromaDBAdapter:
    def __init__(self, collection_name="vector_store"):
        self.client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def insert(self, vectors, metadata):
        ids = [str(m["id"]) for m in metadata]
        metadatas = [{"label": m["label"]} for m in metadata]
        self.collection.add(embeddings=vectors, metadatas=metadatas, ids=ids)

    def search(self, query_vector, top_k=10):
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        return results

    def delete(self, vector_id):
        self.collection.delete(ids=[str(vector_id)])
