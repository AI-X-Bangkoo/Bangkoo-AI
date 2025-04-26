import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from mongo_manager import mongo_manager

"""
최초 작성자: 김동규
최초 작성일: 2025-04-11

- 이미지 및 색상 후처리 함수
"""
# --- 색상 키워드 가져오기 ---
def get_color_keywords_from_db():
    if not mongo_manager.ready:
        mongo_manager.connect()
    db = mongo_manager.db
    doc = db["color_keywords"].find_one({"_id": "korean"})
    if not doc or "dict" not in doc:
        raise ValueError("color_keywords 문서가 없거나 잘못됨")
    return doc["dict"]

# --- 색상 키 추출 ---
def extract_color_token(text):
    color_dict = get_color_keywords_from_db()
    tokens = re.findall(r"[\uac00-\ud7a3a-zA-Z]+", text.lower())
    for color_key, synonyms in color_dict.items():
        if any(token in synonyms for token in tokens):
            return color_key
    return None

# --- 색상 보너스/패널티 적용 ---
def apply_color_bonus(results, color_key, bonus=0.05, penalty=-0.03):
    if not color_key:
        # print("[COLOR] 색상 키 없음 → 보정 생략")
        return results

    color_dict = get_color_keywords_from_db()
    synonyms = color_dict.get(color_key, [])
    # print(f"[COLOR] '{color_key}' 동의어들: {synonyms}")

    for doc in results:
        text = f"{doc.get('name', '')} {doc.get('description', '')} {doc.get('detail', '')}".lower()
        tokens = re.findall(r"[\uac00-\ud7a3a-zA-Z]+", text)
        if any(token in synonyms for token in tokens):
            doc["score"] = doc.get("score", 0) + bonus
            # print(f"[COLOR] 보너스 적용 → {doc.get('name')}")
        else:
            doc["score"] = doc.get("score", 0) + penalty
            # print(f"[COLOR] 패널티 적용 → {doc.get('name')}")
    return results
