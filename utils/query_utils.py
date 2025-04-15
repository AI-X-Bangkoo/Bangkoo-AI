import numpy as np
import torch
from model_loader import model_manager
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from mongo_manager import mongo_manager
from typing import Optional
import re

"""
최초 작성자: 김동규
최초 작성일: 2025-04-11

- 쿼리 전처리 및 확장, 임베딩 생성, 키워드 보너스 계산 유틸리티 함수 모음
- hybrid_search 등 검색 기능에서 공통적으로 활용됨
"""

def infer_category(query: str, db):
    category_keywords_doc = db["category_keywords"].find_one({"_id": "korean"})
    if not category_keywords_doc:
        return None

    category_keywords = category_keywords_doc["dict"]
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in query:
                return category
    return None


def expand_query(query, synonyms):
    words = query.split()
    expanded = []
    for word in words:
        if word in synonyms:
            expanded.append([word] + synonyms[word])
        else:
            expanded.append([word])
    candidates = [' '.join(combo) for combo in product(*expanded)]
    return list(set([query] + candidates))


def get_text_embedding(text):
    text_model = model_manager.text_model
    return text_model.encode([f"query: {text}"], normalize_embeddings=True)


# def get_clip_text_embedding(text):
#     clip_model = model_manager.clip_model
#     clip_processor = model_manager.clip_processor
#     device = model_manager.device

#     inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         features = clip_model.get_text_features(**inputs)
#         features = features / features.norm(dim=-1, keepdim=True)
#     return features.cpu().numpy()
def get_clip_text_embedding(text):
    clip_model = model_manager.clip_model
    clip_processor = model_manager.clip_processor
    device = model_manager.device

    if not text or not text.strip():
        raise ValueError(f"[ERROR] get_clip_text_embedding: 빈 문자열 또는 None 입력: '{text}'")

    try:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        if "input_ids" not in inputs or inputs["input_ids"] is None:
            raise ValueError(f"[ERROR] CLIP 토크나이저 처리 실패: '{text}' → input_ids 없음")

        input_len = inputs["input_ids"].shape[1]
        if input_len < 4:
            raise ValueError(f"[ERROR] 입력 토큰 길이 너무 짧음 (길이={input_len}): '{text}'")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    except Exception as e:
        raise ValueError(f"[ERROR] get_clip_text_embedding 예외 발생: '{text}' → {e}")



def compute_keyword_bonus(product, keywords):
    """
    제품의 name, description, detail 필드에서 쿼리 키워드가 얼마나 많이 매칭되는지를 평가
    반환 값은 0과 1 사이의 값으로, 1에 가까울수록 모든 키워드가 포함되어 있다는 의미
    """
    text = f"{product.get('name','')} {product.get('description','')} {product.get('detail','')}"
    matched = sum(1 for k in keywords if k in text)
    return matched / len(keywords) if keywords else 0

# def extract_color_from_caption(caption: str) -> str:
#     """
#     이미지 캡션에서 색상 정보를 추출하여 color_keywords에 정의된 색상 중 가장 일치하는 색상 반환
#     """
#     if not mongo_manager.ready:
#         mongo_manager.connect()
#     db = mongo_manager.db
#     color_doc = db["color_keywords"].find_one({"_id": "korean"})
#     if not color_doc or "dict" not in color_doc:
#         return None

#     color_dict = color_doc["dict"]
#     caption_lower = caption.lower()
#     for color, keywords in color_dict.items():
#         if any(kw.lower() in caption_lower for kw in keywords):
#             return color
#     return None
def extract_color_from_caption(caption: str) -> str:
    """
    이미지 캡션에서 등장 순서를 기준으로 가장 먼저 등장한 색상 키를 반환
    """
    if not mongo_manager.ready:
        mongo_manager.connect()
    db = mongo_manager.db
    color_doc = db["color_keywords"].find_one({"_id": "korean"})
    if not color_doc or "dict" not in color_doc:
        return None

    color_dict = color_doc["dict"]
    caption_lower = caption.lower()

    best_match = None
    best_index = len(caption_lower)

    for color, keywords in color_dict.items():
        for kw in keywords:
            idx = caption_lower.find(kw.lower())
            if idx != -1 and idx < best_index:
                best_index = idx
                best_match = color

    return best_match

def is_valid_query(query: str) -> bool:
    return query is not None and query.strip() != ""

def extract_keywords_from_query(query: str, db):
    color_dict = db["color_keywords"].find_one({"_id": "korean"})["dict"]
    category_dict = db["category_keywords"].find_one({"_id": "korean"})["dict"]

    detected_color = None
    detected_category = None

    for color, keywords in color_dict.items():
        if any(k in query for k in keywords):
            detected_color = color
            break

    for category, keywords in category_dict.items():
        if any(k in query for k in keywords):
            detected_category = category
            break

    return detected_color, detected_category

def separate_korean_words(query: str) -> str:
    import re
    return re.sub(r'(?<=[가-힣])(?=[가-힣])', ' ', query)

def get_shape_keywords_from_db():
    db = mongo_manager.db
    doc = db["shape_keywords"].find_one({"_id": "korean"})
    return doc["dict"] if doc and "dict" in doc else {}

def extract_shape_from_caption(caption: str, db):
    shape_dict = db["shape_keywords"].find_one({"_id": "korean"})
    if not shape_dict or "dict" not in shape_dict:
        return None, []

    caption = caption.lower()
    match_key = None
    synonyms = []
    for shape_key, keywords in shape_dict["dict"].items():
        for word in keywords:
            if word in caption:
                match_key = shape_key
                synonyms = keywords
                print(f"[SHAPE] 형태 키 '{shape_key}' 추출됨 (매칭: {word})")
                return match_key, synonyms
    print("[SHAPE] 형태 키 없음")
    return None, []

# --- 공백 삽입 함수 ---
def auto_insert_space(query: str, db) -> str:
    color_doc = db["color_keywords"].find_one({"_id": "korean"})
    shape_doc = db["shape_keywords"].find_one({"_id": "korean"})
    category_doc = db["category_keywords"].find_one({"_id": "korean"})

    color_keywords = [item for sublist in color_doc["dict"].values() for item in sublist]
    shape_keywords = [item for sublist in shape_doc["dict"].values() for item in sublist]
    category_keywords = [item for sublist in category_doc["dict"].values() for item in sublist]

    parts = []
    temp = query
    for kw_list in [color_keywords, shape_keywords, category_keywords]:
        for kw in sorted(kw_list, key=len, reverse=True):  # 긴 단어 우선
            if kw in temp:
                parts.append(kw)
                temp = temp.replace(kw, " ", 1)  # 첫 등장만 제거
                break  # 한 분류당 하나만 추출

    parts.append(temp.replace(" ", ""))  # 남은 부분
    return " ".join(filter(None, parts))

