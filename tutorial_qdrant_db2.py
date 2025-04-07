from qdrant_client import AsyncQdrantClient, models
import numpy as np
import asyncio

async def main():
   # Your async code using QdrantClient might be put here
   client = AsyncQdrantClient(url="http://localhost:6333")

   if not await client.collection_exists("my_collection"):
      await client.create_collection(
         collection_name="my_collection",
         vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
      )

   await client.upsert(
      collection_name="my_collection",
      points=[
            models.PointStruct(
               id=i,
               vector=np.random.rand(10).tolist(),
            )
            for i in range(100)
      ],
   )

   res = await client.search(
      collection_name="my_collection",
      query_vector=np.random.rand(10).tolist(),  # type: ignore
      limit=10,
   )

   print(res)

asyncio.run(main())