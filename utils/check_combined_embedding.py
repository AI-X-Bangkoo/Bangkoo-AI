import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo_manager import mongo_manager 
import numpy as np

# Mongo 연결
if not mongo_manager.ready:
    mongo_manager.connect()
collection = mongo_manager.products

# 랜덤하게 문서 3개 조회
sample_docs = collection.aggregate([
    {"$match": {"combinedEmbedding": {"$type": "array"}}},
    {"$sample": {"size": 3}}
])

print("\n✅ combinedEmbedding 검사 시작...\n")
for i, doc in enumerate(sample_docs, 1):
    vec = np.array(doc["combinedEmbedding"], dtype=np.float32)
    dim = vec.shape[0]
    norm = np.linalg.norm(vec)

    print(f"[{i}] _id: {doc['_id']}")
    print(f"   ➤ 차원: {dim}")
    print(f"   ➤ 벡터 norm: {norm:.6f}")
    print(f"   ➤ 정규화 상태: {'✅ OK' if np.isclose(norm, 1.0, atol=1e-4) else '⚠️ NOT NORMALIZED'}")
    print()
