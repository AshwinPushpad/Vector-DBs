from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Check if collection exists before creating

if connections.has_connection("person_info"):
    collection = Collection("person_info")  # Load existing collection
else:
    # Define schema

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="name_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="age_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="job_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    ]
    schema = CollectionSchema(fields, description="Person Information")

    # Create collection
    collection = Collection("person_info")
    collection.drop()
    collection = Collection("person_info", schema)

from Application.embeddings import OpenAIEmbedding
model = OpenAIEmbedding()


# Insert Data - Column-wise format
data = [
    # [None],  # Auto-generated ID
    [model.encode("Ashwin")],  # Name vector
    [model.encode("22 years old")],  # Age vector
    [model.encode("Engineer")],  # Job vector
]

collection.insert(data)

# ✅ Create index on the correct fields
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
for field in ["name_vector", "age_vector", "job_vector"]:  # ✅ Correct field names
    collection.create_index(field, index_params)

# ✅ Load collection before searching
collection.load()

# Query
query_vector = model.encode("What is Ashwin’s age?")

# Perform hybrid search over ALL vector fields
search_results = []
for field in ["name_vector", "age_vector", "job_vector"]:
    results = collection.search(
        data=[query_vector],
        anns_field=field,  # Search in each vector field
        param={"metric_type": "L2", "nprobe": 10},
        limit=1,
        output_fields=["id", "name_vector", "age_vector", "job_vector"]
    )
    search_results.extend(results)  # ✅ Fix: Extend without indexing results[0]

print(len(search_results))
# model.
# Format Results
# retrieved_info = {"id": None, "name": None, "age": None, "job": None}
# for hit in search_results:
#     # retrieved_info["id"] = hit.id
#     retrieved_info["name"] = hit.entity["name_vector"]  # ✅ Fix: Use direct access
#     retrieved_info["age"] = hit.entity["age_vector"]
#     retrieved_info["job"] = hit.entity["job_vector"]

# print(f"Retrieved Data for RAG: {retrieved_info}")
