import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo_manager import mongo_manager
mongo_manager.connect()
from api.search.hybrid_search import tokenize_clean

def patch_indexed_tokens():
    db = mongo_manager.db
    products = db["products"]

    cursor = products.find(
        {"indexedTokens": {"$exists": False}},
        {"_id": 1, "name": 1, "description": 1, "detail": 1}
    )

    updates = 0
    for doc in cursor:
        text = f"{doc.get('name', '')} {doc.get('description', '')} {doc.get('detail', '')}"
        tokens = tokenize_clean(text)
        products.update_one(
            {"_id": doc["_id"]},
            {"$set": {"indexedTokens": tokens}}
        )
        updates += 1
        if updates % 100 == 0:
            print(f"{updates}개 문서 패치 완료")

    print(f"[완료] 총 {updates}개 문서에 indexedTokens 추가 완료")

if __name__ == "__main__":
    mongo_manager.connect()  # 반드시 먼저 연결
    patch_indexed_tokens()