import os
import sys
import numpy as np
from tqdm import tqdm

"""
최초 작성자: 김동규
최초 작성일: 2025-04-10

- 이미지 임베딩(1024차원) + 텍스트 임베딩(768차원)을 결합하여 
  벡터 검색용 combinedEmbedding(1792차원) 필드 생성

- 기존 combinedEmbedding 필드 제거
- imageEmbedding과 textEmbedding을 정규화 후 0.7:0.3 비율로 결합
- 결합된 벡터도 정규화 후 float32 배열 형태로 MongoDB에 저장
- 벡터 검색(Index type: float32 array)에 사용 가능
"""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo_manager import mongo_manager

BATCH_SIZE = 500

# MongoDB 연결
if not mongo_manager.ready:
    mongo_manager.connect()
collection = mongo_manager.products

# Step 1: combinedEmbedding 필드 삭제
print("기존 combinedEmbedding 필드 삭제 중...")
delete_result = collection.update_many(
    {"combinedEmbedding": {"$exists": True}},
    {"$unset": {"combinedEmbedding": ""}}
)
print(f"삭제 완료: {delete_result.modified_count}개 문서\n")

# Step 2: ID만 미리 가져오기
query = {
    "imageEmbedding": {"$exists": True},
    "textEmbedding": {"$exists": True}
}
ids = [doc["_id"] for doc in collection.find(query, {"_id": 1}).sort("_id", 1)]
total = len(ids)
print(f"총 대상 문서 수: {total}개")

count = 0
for i in range(0, total, BATCH_SIZE):
    batch_ids = ids[i:i + BATCH_SIZE]
    docs = collection.find({"_id": {"$in": batch_ids}}).batch_size(50)
    
    for doc in tqdm(docs, desc=f"[{i}~{i+len(batch_ids)}] 처리 중"):
        try:
            image_emb = np.array(doc["imageEmbedding"], dtype=np.float32)
            text_emb = np.array(doc["textEmbedding"], dtype=np.float32)

            if image_emb.shape[0] != 1024 or text_emb.shape[0] != 768:
                continue

            image_emb /= np.linalg.norm(image_emb)
            text_emb /= np.linalg.norm(text_emb)
            combined = np.concatenate([image_emb * 0.7, text_emb * 0.3])
            combined /= np.linalg.norm(combined)

            # float 배열로 저장
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"combinedEmbedding": combined.tolist()}}
            )
            count += 1

        except Exception as e:
            print(f"오류 (ID: {doc['_id']}): {e}")
            continue

print(f"\n최종 완료: 총 {count}개 문서 updated (배치 단위, float 배열 저장)")
