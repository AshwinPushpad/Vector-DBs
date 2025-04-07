from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

connections.connect("default", host="localhost", port="19530")

fields = [
    # , auto_id=True
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="name_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="age_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="job_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
]

schema = CollectionSchema(fields, description="Person Information")
collection = Collection('person_info')
collection.drop()
collection = Collection("person_info", schema)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

data = [
    [1],  # ID
    [model.encode("Ashwin").tolist()],  # Name vector
    [model.encode("22 years old").tolist()],  # Age vector
    [model.encode("Engineer").tolist()],  # Job vector
]

collection.insert(data)

query_vector = model.encode("What is Ashwin’s age?").tolist()

# ✅ Create index on the correct fields
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
for field in ["name_vector", "age_vector", "job_vector"]:  # ✅ Correct field names
    collection.create_index(field, index_params)
collection.load()
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
    search_results.extend(results)

print(search_results) # # Combine retrieved results into structured data for RAG
# retrieved_info = {}
# for hit in search_results:
#     retrieved_info["id"] = hit.id
#     retrieved_info["name"] = hit.entity.get("name_vector")
#     retrieved_info["age"] = hit.entity.get("age_vector")
#     retrieved_info["job"] = hit.entity.get("job_vector")

# print(f"Retrieved Data for RAG: {retrieved_info}")

# from openai import OpenAI

# # Convert vectors back to text (optional, if needed)
# retrieved_text = model.decode(retrieved_info["age"])

# # Send structured data to OpenAI for answer generation
# response = openai.ChatCompletion.create(
#     model="gpt-4-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": f"What is Ashwin’s age? Here’s the data: {retrieved_text}"}
#     ]
# )

# print(response["choices"][0]["message"]["content"])
