import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo_manager import mongo_manager

if not mongo_manager.ready:
    mongo_manager.connect()
collection = mongo_manager.products

docs = collection.find({"combinedEmbedding": {"$exists": True}})
invalid_ids = []

for doc in tqdm(docs, desc="🔍 벡터 유효성 검사 중"):
    vec = doc["combinedEmbedding"]
    if not isinstance(vec, list) or len(vec) != 1792:
        invalid_ids.append(doc["_id"])
        continue
    arr = np.array(vec, dtype=np.float32)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        invalid_ids.append(doc["_id"])

print(f"\n❌ 유효하지 않은 벡터 문서 수: {len(invalid_ids)}")

# 필요 시 삭제
if len(invalid_ids):
    confirm = input("🧹 삭제할까요? (y/n): ").strip().lower()
    if confirm == "y":
        result = collection.delete_many({"_id": {"$in": invalid_ids}})
        print(f"✅ 삭제 완료: {result.deleted_count}개 문서 삭제됨")
    else:
        print("⏹ 삭제하지 않음")
else:
    print("✅ 모든 벡터가 유효합니다")
