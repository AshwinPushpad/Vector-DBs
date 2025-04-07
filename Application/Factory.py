from Adapters.chroma_adapter import ChromaDBAdapter
from Adapters.milvus_adapter import MilvusDBAdapter
from Adapters.qdrant_adapter import QdrantDBAdapter

class VectorDB:
    def __init__(self, db_type="chromadb", **kwargs):
        if db_type == "chromadb":
            self.db = ChromaDBAdapter(**kwargs)
        elif db_type == "milvus":
            self.db = MilvusDBAdapter(**kwargs)
        elif db_type == "qdrant":
            self.db = QdrantDBAdapter(**kwargs)
        else:
            raise ValueError("Unsupported database type")

    def insert(self, vectors, metadata):
        return self.db.insert(vectors, metadata)

    def search(self, query_vector, top_k=10):
        return self.db.search(query_vector, top_k)

    def delete(self, vector_id):
        return self.db.delete(vector_id)
