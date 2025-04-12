import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from mongo_manager import mongo_manager
import math

"""
최초 작성자: 김동규
최초 작성일: 2025-04-11

- 이미지 및 색상 후처리 함수
"""

# --- 이미지 유사도 재랭킹 ---
# def rerank_by_visual_similarity(results, query_image_embedding, weight=0.3):
#     # (1, 1024) 보장
#     if isinstance(query_image_embedding, np.ndarray) and query_image_embedding.ndim == 1:
#         query_image_embedding = query_image_embedding.reshape(1, -1)

#     valid_docs = []
#     valid_vectors = []

#     for doc in results:
#         vec = doc.get("imageEmbedding")
#         if isinstance(vec, list) and len(vec) == 1024 and all(isinstance(v, (int, float)) for v in vec) and not any(np.isnan(v) for v in vec):
#             valid_docs.append(doc)
#             valid_vectors.append(vec)
#         else:
#             print(f"[RERANK] 제외된 제품: {doc.get('name')} (imageEmbedding 문제) → 타입: {type(vec)}, 길이: {len(vec) if isinstance(vec, list) else 'N/A'}, 값 예시: {str(vec)[:60]}")

#     if not valid_vectors:
#         print("[RERANK] 유효한 이미지 임베딩 없음 → 스킵")
#         return results

#     image_vectors = np.array(valid_vectors, dtype=np.float32)

#     try:
#         sim_scores = cosine_similarity(query_image_embedding, image_vectors)[0]
#         print("[RERANK] 이미지 기반 유사도:", sim_scores[:5])
#     except Exception as e:
#         print(f"[RERANK] cosine_similarity 오류: {e}")
#         return results

#     for i, doc in enumerate(valid_docs):
#         doc["score"] = (1 - weight) * doc.get("score", 0) + weight * sim_scores[i]

#     all_docs = []
#     for doc in results:
#         if doc in valid_docs:
#             all_docs.append(doc)
#         else:
#             doc["score"] = doc.get("score", 0)
#             all_docs.append(doc)

#     return sorted(all_docs, key=lambda x: x.get("score", 0), reverse=True)

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
        print("[COLOR] 색상 키 없음 → 보정 생략")
        return results

    color_dict = get_color_keywords_from_db()
    synonyms = color_dict.get(color_key, [])
    print(f"[COLOR] '{color_key}' 동의어들: {synonyms}")

    for doc in results:
        text = f"{doc.get('name', '')} {doc.get('description', '')} {doc.get('detail', '')}".lower()
        tokens = re.findall(r"[\uac00-\ud7a3a-zA-Z]+", text)
        if any(token in synonyms for token in tokens):
            doc["score"] = doc.get("score", 0) + bonus
            print(f"[COLOR] 보너스 적용 → {doc.get('name')}")
        else:
            doc["score"] = doc.get("score", 0) + penalty
            print(f"[COLOR] 패널티 적용 → {doc.get('name')}")
    return results
# def apply_color_bonus(results, color_key):
#     if not color_key:
#         print("[COLOR] 색상 키 없음 → 보정 생략")
#         return results

#     color_dict = get_color_keywords_from_db()
#     synonyms = color_dict.get(color_key, [])
#     print(f"[COLOR] '{color_key}' 동의어들: {synonyms}")

#     COLOR_BONUS_FACTOR = 1.2
#     COLOR_PENALTY_FACTOR = 0.7

#     for doc in results:
#         try:
#             score = doc.get("score", None)
#             if score is None:
#                 # 기존 final_scores 에서 score 필드가 없을 수 있으므로 기본 0점 설정
#                 score = doc.get("유사도", 0)  # 혹은 0

#             text = f"{doc.get('name', '')} {doc.get('description', '')} {doc.get('detail', '')}".lower()
#             if any(syn in text for syn in synonyms):
#                 doc["score"] = score * COLOR_BONUS_FACTOR
#                 print(f"[COLOR] 보너스 적용 → {doc.get('name')}")
#             else:
#                 doc["score"] = score * COLOR_PENALTY_FACTOR
#                 print(f"[COLOR] 패널티 적용 → {doc.get('name')}")
#         except Exception as e:
#             print(f"[COLOR] 색상 처리 실패: {e}")
#             doc["score"] = score  # 예외 시 기존 점수 유지

#     return results
