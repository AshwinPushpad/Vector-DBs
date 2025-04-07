import weaviate

# client = weaviate.Client("https://edu-demo.weaviate.network")
# print(client.is_ready())  # Should print: True
# Connect to Weaviate server

client = weaviate.WeaviateClient(
    connection=weaviate.connect_to_local()
)
# Define Schema
schema = {
    "class": "FakeDocuments",
    "vectorIndexType": "hnsw",
    "properties": [
        {"name": "name", "dataType": ["string"]},
        {"name": "text", "dataType": ["string"]}
    ]
}

# Create Collection
client.schema.create_class(schema)
print("Collection Created!")


from faker import Faker

fake = Faker()

with client.batch as batch:
    for _ in range(1000000):  # 1M Fake Documents
        batch.add_data_object(
            {"name": fake.name(), "text": fake.paragraph()},
            class_name="FakeDocuments"
        )

print("Data Inserted Successfully!")


query = client.query.get("FakeDocuments", ["name", "text"]).with_near_text({"concepts": ["AI research"]}).do()
print(query)
