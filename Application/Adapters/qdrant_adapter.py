from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class QdrantDBAdapter:
    def __init__(self, collection_name="vector_store", dim=128):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name

        # Create collection if not exists
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def insert(self, vectors, metadata):
        points = [
            PointStruct(id=m["id"], vector=v, payload={"label": m["label"]})
            for v, m in zip(vectors, metadata)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, top_k=10):
        results = self.client.search(self.collection_name, query_vector, top_k=top_k)
        return results

    def delete(self, vector_id):
        self.client.delete(collection_name=self.collection_name, point_ids=[vector_id])
