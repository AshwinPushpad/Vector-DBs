from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

class MilvusDBAdapter:
    def __init__(self, collection_name="vector_store", dim=128):
        connections.connect("default", host="localhost", port="19530")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields)
        self.collection = Collection(name=collection_name, schema=schema)
        self.collection.load()

    def insert(self, vectors, metadata):
        ids = [m["id"] for m in metadata]
        data = [[i] for i in ids] + [vectors]
        self.collection.insert(data)

    def search(self, query_vector, top_k=10):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search([query_vector], "vector", search_params, top_k)
        return results

    def delete(self, vector_id):
        expr = f"id == {vector_id}"
        self.collection.delete(expr)
