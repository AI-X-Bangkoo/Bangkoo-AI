# migrations/update_indexedTokens.py
import os
from pymongo import MongoClient, UpdateOne
from konlpy.tag import Okt

# MongoDB 연결
client = MongoClient(os.getenv("MONGO_URI"))
db     = client.get_default_database()
prods  = db["products"]

okt = Okt()

# 전체 문서 순회하며 preprocessedText, indexedTokens 생성
# 2) bulk ops 리스트 준비
bulk_ops = []

# 3) 미리 preprocessedText, indexedTokens 컬럼이 없거나 업데이트가 필요한 문서를 순회
for doc in prods.find({}, {"name":1,"description":1,"detail":1}):
    # 원본 텍스트 합치기
    text_full = " ".join(filter(None, [
        doc.get("name",""),
        doc.get("description",""),
        doc.get("detail","")
    ]))
    # ① 소문자 정규화
    preprocessed = text_full.lower()
    # ② Okt로 형태소 분석
    tokens = okt.morphs(preprocessed)

    # ③ UpdateOne 모델 생성
    bulk_ops.append(
        UpdateOne(
            {"_id": doc["_id"]},
            {"$set": {
                "preprocessedText": preprocessed,
                "indexedTokens":      tokens
            }}
        )
    )

# 4) 실제로 bulk_write 실행 (매 500개씩 나눠서)
BATCH_SIZE = 500
if not bulk_ops:
    print("▶️ 업데이트할 문서가 없습니다.")
else:
    total_modified = 0
    for i in range(0, len(bulk_ops), BATCH_SIZE):
        batch = bulk_ops[i : i + BATCH_SIZE]
        result = prods.bulk_write(batch, ordered=False)
        total_modified += result.modified_count
        print(f"  └ batch {i//BATCH_SIZE+1}: matched={result.matched_count}, modified={result.modified_count}")

    print(f"✅ 전체 수정된 문서 수: {total_modified}")