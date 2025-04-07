import os
import io
import numpy as np
from PIL import Image
import torch
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from model_loader import model_manager

load_dotenv()

"""
최초 작성자: 김동규
최초 작성일: 2025-04-07

하이브리드 검색 모듈 (안정화 버전)
- 모델이 로드되지 않았을 경우 예외 처리
- 모델은 함수 내에서 동적으로 접근
- DB 연결도 함수 내에서 처리
"""

def expand_query(query, synonyms):
    words = query.split()
    expanded = []
    for word in words:
        if word in synonyms:
            expanded.append([word] + synonyms[word])
        else:
            expanded.append([word])
    from itertools import product
    candidates = [' '.join(combo) for combo in product(*expanded)]
    return list(set([query] + candidates))

def get_text_embedding(text):
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    text_model = model_manager.text_model
    return text_model.encode([f"query: {text}"], normalize_embeddings=True)

def get_clip_text_embedding(text):
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    clip_model = model_manager.clip_model
    clip_processor = model_manager.clip_processor
    device = model_manager.device

    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def hybrid_search(query, top_k=10):
    print("[DEBUG] hybrid_search 진입")
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")

    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    print("[DEBUG] MONGO_URI =", MONGO_URI)
    db = client["bangkoo"]
    product_collection = db["products"]
    print("[DEBUG] Mongo 연결 성공")
    synonyms_doc = db["synonyms"].find_one({"_id": "korean"})
    print("[DEBUG] synonyms_doc:", synonyms_doc)

    if synonyms_doc is None or "dict" not in synonyms_doc:
        raise ValueError("동의어 사전을 찾을 수 없습니다.")

    synonyms = synonyms_doc["dict"]
    products = list(product_collection.find())
    items = products
    image_embeddings = np.array([p["imageEmbedding"] for p in products], dtype=np.float32)
    text_embeddings = np.array([p["textEmbedding"] for p in products], dtype=np.float32)
    print(f"image_embeddings: {image_embeddings}")
    print(f"text_embeddings: {text_embeddings}")

    queries = expand_query(query, synonyms)
    print(f"동의어 확장 결과: {queries}")

    best_score = -1
    best_indices = []
    best_sim = None

    for q in queries:
        e5_embed = get_text_embedding(q)
        clip_embed = get_clip_text_embedding(q)

        sim_text = cosine_similarity(e5_embed, text_embeddings)[0]
        sim_image = cosine_similarity(clip_embed, image_embeddings)[0]
        sim = 0.6 * sim_text + 0.4 * sim_image

        top_idx = np.argsort(sim)[::-1]
        if sim[top_idx[0]] > best_score:
            best_score = sim[top_idx[0]]
            best_indices = top_idx
            best_sim = sim

    results = []
    for i in best_indices[:top_k]:
        item = items[i]
        results.append({
            "이름": item["name"],
            "설명": item["description"],
            "상세설명": item.get("detail", ""),
            "링크": item["link"],
            "이미지": item["imageUrl"],
            "할인가": item.get("price", "정보 없음"),
            "정상가": item.get("price", "정보 없음"),
            "카테고리": item.get("category"),
            "csv": item.get("csv", ""),
            "유사도": float(best_sim[i])
        })

    return results