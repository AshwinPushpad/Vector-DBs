from pymilvus import MilvusClient, Collection, CollectionSchema, FieldSchema, DataType, connections

# client = MilvusClient(uri="http://localhost:19530")
# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

Collection('user_data').drop()
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=6),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="metadatas", dtype=DataType.JSON, max_length=100)
]

# Create collection schema
schema = CollectionSchema(fields)

# if client.has_collection(collection_name="user_data"):
# collection = Collection('user_data')
#     # client.drop_collection(collection_name="user_data")
#     pass
# else:
#     # client.create_collection(
#     #     collection_name="user_data",
#     #     dimension=1536
#     #     schema
#         # )
collection = Collection('user_data',schema)
collection.flush()

collection.create_index(field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1536}})
collection.load()
# print(collection.num_entities)
from embeddings import OpenAIEmbedding
model = OpenAIEmbedding()

def save_data(data,id=f"id{collection.num_entities+1}"):
    # print(id)
    id=id 
    # print(id)
    vectors = model.encode(str(data))
    metadatas={'id':id,'data':data}

    data = [[id],[vectors],[metadatas]]

    try:
        collection.upsert(data)
        # print(collection.get(id))
    except ValueError as ve:
        print(ve)
        return ve
    else:
        return "data saved successfully...."
    finally:
        collection.flush()


def query_data(query, top_k=2):
    collection.flush()
    query_vectors = model.encode(query)

    search_results = collection.search(
        data=[query_vectors],  # List of query vectors
        anns_field="vector",  # The field to search on
        param={"metric_type": "L2", "params": {"nprobe": 20}},  # Distance metric & search params
        limit=top_k,  # Return top_k closest vectors
        output_fields=["metadatas"]  # Include metadata in the result
        )
    # print(search_results)
    return [hit.entity.metadatas for hits in list(search_results) for hit in list(hits)]
    # return [res for res in search_results["metadatas"]]

# res = client.delete(
#     collection_name="user_data",
#     filter="subject == 'history'",
# )
# print(res)

# res= store_data({"name": "John Doe", "location": "London", "interests": ["reading", "hiking"]})
# print(res)

# res= save_data({"name": "John doe"})
# print('res: ',res)
# res = query_data("John cena")
# print(res)



# all_data = collection.query(
#     expr="",  # Match all IDs (assuming IDs are non-negative)
#     limit = 10,
#     output_fields=["metadatas"]  # Include required fields
# )
# # Print results
# for record in all_data:
#     print(record)

# collection.flush()
# print(collection.num_entities)
