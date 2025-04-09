from mongo_manager import mongo_manager
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 500

# MongoDB 연결
if not mongo_manager.ready:
    mongo_manager.connect()
collection = mongo_manager.products

# 대상 문서 ID만 미리 가져오기
query = {
    "imageEmbedding": {"$type": "array"},
    "textEmbedding": {"$type": "array"}
}
all_ids = [doc["_id"] for doc in collection.find(query, {"_id": 1}).sort("_id", 1)]
total = len(all_ids)
print(f"총 문서 수: {total}")

count = 0
for i in range(0, total, BATCH_SIZE):
    batch_ids = all_ids[i:i + BATCH_SIZE]
    documents = collection.find({"_id": {"$in": batch_ids}})

    for doc in tqdm(documents, desc=f"{i} ~ {i+BATCH_SIZE} 처리 중"):
        try:
            image_emb = np.array(doc["imageEmbedding"], dtype=np.float32)
            text_emb = np.array(doc["textEmbedding"], dtype=np.float32)

            if image_emb.shape[0] != 1024 or text_emb.shape[0] != 768:
                continue

            # 정규화 + 결합 + 재정규화
            image_emb /= np.linalg.norm(image_emb)
            text_emb /= np.linalg.norm(text_emb)
            combined = np.concatenate([image_emb, text_emb])
            combined /= np.linalg.norm(combined)

            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"combinedEmbedding": combined.tolist()}}
            )
            count += 1

        except Exception as e:
            print(f"오류 (ID: {doc['_id']}): {e}")
            continue

print(f"\n최종 완료: {count}개 문서 updated")
