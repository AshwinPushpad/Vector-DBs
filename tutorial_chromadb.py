import chromadb

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
# client = chromadb.Client()
client = chromadb.PersistentClient(path="./chroma_db")  

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
# collection = client.create_collection("all-my-documents")

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = client.get_or_create_collection(name="my_collection")
# collection = client.get_collection("my_collection")
# collection = client.create_collection("my_collection")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
# collection.add(
#     documents=["This is document1", "This is document2"], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
#     metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
#     ids=["doc1", "doc2"], # unique for each doc
# )
# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about dc",
        "This is a document about pixar",
        "This is a document about anime",
        "This is a document about marvel",
    ],
    ids=["id11", "id21","id13","id41"]
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document about all-might"], # Chroma will embed this for you
    n_results=2 # how many results to return
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)
print(results)

# print(collection.count())
# print(collection.get())