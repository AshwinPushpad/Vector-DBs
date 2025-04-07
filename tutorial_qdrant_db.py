from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

from qdrant_client.models import VectorParams, Distance

if not client.collection_exists("my_collection"):
   client.create_collection(
      collection_name="my_collection",
      vectors_config=VectorParams(size=100, distance=Distance.COSINE),
   )

import numpy as np
from qdrant_client.models import PointStruct

vectors = np.random.rand(100, 100)
client.upsert(
   collection_name="my_collection",
   points=[
      PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
      )
      for idx, vector in enumerate(vectors)
   ]
)

# q_vector = np.random.rand(100)
hits = client.query_points(
   collection_name="my_collection",
   # query_vector=q_vector,
   limit=5  # Return 5 closest points
)

from qdrant_client.models import Filter, FieldCondition, Range

hits = client.query_points(
   collection_name="my_collection",
   # query_vector=q_vector,
   query_filter=Filter(
      must=[  # These conditions are required for search results
            FieldCondition(
               key='rand_number',  # Condition based on values of `rand_number` field.
               range=Range(
                  gte=3  # Select only those results where `rand_number` >= 3
               )
            )
      ]
   ),
   limit=5  # Return 5 closest points
)

print(hits)