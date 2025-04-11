import os
import sys
import numpy as np
from tqdm import tqdm

# 경로 설정 및 mongo_manager import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo_manager import mongo_manager

BATCH_SIZE = 500

# MongoDB 연결
if not mongo_manager.ready:
    mongo_manager.connect()
collection = mongo_manager.products

# ✅ Step 1: combinedEmbedding 필드 삭제
print("🧹 기존 combinedEmbedding 필드 삭제 중...")
delete_result = collection.update_many(
    {"combinedEmbedding": {"$exists": True}},
    {"$unset": {"combinedEmbedding": ""}}
)
print(f"🧼 삭제 완료: {delete_result.modified_count}개 문서\n")

# ✅ Step 2: ID만 미리 가져오기
query = {
    "imageEmbedding": {"$exists": True},
    "textEmbedding": {"$exists": True}
}
ids = [doc["_id"] for doc in collection.find(query, {"_id": 1}).sort("_id", 1)]
total = len(ids)
print(f"📦 총 대상 문서 수: {total}개")

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
            print(f"❌ 오류 (ID: {doc['_id']}): {e}")
            continue

print(f"\n✅ 최종 완료: 총 {count}개 문서 updated (배치 단위, float 배열 저장)")
