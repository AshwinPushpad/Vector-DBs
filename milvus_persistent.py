from pymilvus import connections

# Connect to the persistent Milvus instance
connections.connect("default", host="localhost", port="19530")

print("✅ Connected to Persistent Milvus!")

# from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
# import random

# # Define schema
# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2)
# ]
# schema = CollectionSchema(fields, description="Persistent collection")
# collection = Collection(name="persistent_collection", schema=schema)

# # Insert some random vectors
# data = [[random.uniform(0, 1) for _ in range(2)] for _ in range(5)]  # 5 vectors, dim=2
# collection.insert([data])  

# print("✅ Data Inserted in Persistent Milvus!")

from pymilvus import utility
print(utility.list_collections())

# Check if the collection still exists
# # has_collection("persistent_collection"):
#     print("✅ Data is still there after restart!")
# else:
#     print("❌ Data was lost!")
