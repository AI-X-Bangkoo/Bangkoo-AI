from pymongo import MongoClient
import numpy as np
import os

client = MongoClient(os.getenv("MONGO_URI"))
collection = client["bangkoo"]["products"]

# 쿼리 벡터 아무거나 테스트용으로 생성
test_vec = np.random.rand(1792).tolist()

cursor = collection.aggregate([
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "combinedEmbedding",
            "queryVector": test_vec,
            "numCandidates": 100,
            "limit": 5,
            "similarity": "cosine"
        }
    }
])

for doc in cursor:
    print(doc.get("name"), doc.get("score"))
