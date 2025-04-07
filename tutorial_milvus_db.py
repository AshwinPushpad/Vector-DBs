from pymilvus import MilvusClient , connections
import numpy as np

# client = MilvusClient("./milvus_demo.db")

client = MilvusClient(uri="http://localhost:19530")
# client.create_collection(
#     collection_name="demo_collection",
#     dimension=384  # The vectors we will use in this demo has 384 dimensions
# )

if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=1536,  # The vectors we will use in this demo has 768 dimensions
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

from Application.embeddings import OpenAIEmbedding
model = OpenAIEmbedding()

# vectors = [[ np.random.uniform(-1, 1) for _ in range(768) ] for _ in range(len(docs)) ]
data = [ {"id": i, "vector": model.encode(docs[i]), "text": docs[i], "subject": "history"} for i in range(len(docs)) ]
res = client.insert(
    collection_name="demo_collection",
    data=data
)

res = client.search(
    collection_name="demo_collection",
    data=[model.encode("where was Turing born")],
    # filter="subject == 'history'",
    limit=1,
    output_fields=['id', "text", "subject"],
)
print(res)

# res = client.query(
#     collection_name="demo_collection",
#     # filter="subject == 'history'",
#     output_fields=["text", "subject"],
# )
# print(res)

res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'history'",
)
print(res)
