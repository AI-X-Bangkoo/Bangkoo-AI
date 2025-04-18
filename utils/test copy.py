import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# 경로 설정 및 mongo_manager import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo_manager import mongo_manager

# MongoDB 연결
if not mongo_manager.ready:
    mongo_manager.connect()
collection = mongo_manager.products

# 1. 아무거나 두 개 가져오기 (combinedEmbedding 필드 있는 것 중)
docs = list(collection.find({"combinedEmbedding": {"$exists": True}}).limit(2))
if len(docs) < 2:
    print("벡터가 있는 제품이 2개 이상 있어야 합니다.")
    exit()

doc1, doc2 = docs[0], docs[1]

# 2. 벡터 추출 및 유사도 계산
vec1 = np.array(doc1["combinedEmbedding"], dtype=np.float32).reshape(1, -1)
vec2 = np.array(doc2["combinedEmbedding"], dtype=np.float32).reshape(1, -1)

# 정규화
vec1 = vec1 / np.linalg.norm(vec1)
vec2 = vec2 / np.linalg.norm(vec2)

# 유사도
similarity = cosine_similarity(vec1, vec2)[0][0]

print(f"제품 1: {doc1['name']}")
print(f"제품 2: {doc2['name']}")
print("cosine similarity:", similarity)
