import os
import numpy as np
from PIL import Image
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from model_loader import model_manager
import torch

load_dotenv()

"""
최초 작성자: 김동규
최초 작성일: 2025-04-07

하이브리드 검색 모듈 (MongoDB 기반 카테고리 키워드 사용 버전)
- 키워드 기반 필터링
- 카테고리 필터링 (DB에서 동적으로)
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

def hybrid_search(query, top_k=10):  # top_k=10으로 상위 10개만 나오도록 되어있는데 추후 전체 결과 보여주려면 None으로 변경
    print("[DEBUG] hybrid_search 진입")
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")

    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client["bangkoo"]
    product_collection = db["products"]
    print("[DEBUG] Mongo 연결 성공")

    synonyms_doc = db["synonyms"].find_one({"_id": "korean"})
    if synonyms_doc is None or "dict" not in synonyms_doc:
        raise ValueError("동의어 사전을 찾을 수 없습니다.")
    synonyms = synonyms_doc["dict"]

    products = list(product_collection.find())
    
    # 카테고리 필터링 (강제 일치 기반)
    inferred = infer_category(query, db)
    print(f"[DEBUG] inferred 카테고리: {inferred}")
    if inferred:
        products = [p for p in products if inferred == (p.get("category") or "").strip()]
        print(f"[DEBUG] 카테고리 '{inferred}' 필터링 후 개수: {len(products)}")

    # 키워드 필터링
    keywords = query.split()
    keyword_filtered = []
    for p in products:
        text = f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}"
        if all(k in text for k in keywords):
            keyword_filtered.append(p)
    if len(keyword_filtered) >= 5:
        products = keyword_filtered
        print(f"[DEBUG] 키워드 필터 적용됨: {len(products)}개")

    items = products
    image_embeddings = np.array([p["imageEmbedding"] for p in products], dtype=np.float32)
    text_embeddings = np.array([p["textEmbedding"] for p in products], dtype=np.float32)

    queries = expand_query(query, synonyms)
    print(f"[DEBUG] 동의어 확장 결과: {queries}")

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
    
    # 전체 결과 보여주려면 주석 해제
    # if top_k is None:
    #     top_k = len(best_indices) 

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
            "유사도": float(best_sim[i]),
            "추천이유": f"쿼리 '{query}' 와 유사도 {float(best_sim[i]):.3f}"
        })

    # 최종 결과 필터링
    # if inferred:
    #     filtered_results = [r for r in results if inferred == (r.get("카테고리") or "").strip()]
    #     if len(filtered_results) >= top_k:
    #         print(f"[DEBUG] 최종 결과에서 카테고리 필터 적용됨 → {inferred}, 개수: {len(filtered_results)}")
    #         results = filtered_results

    return results