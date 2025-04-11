import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from model_loader import model_manager
import torch
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mongo_manager import mongo_manager

load_dotenv()

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
    text_model = model_manager.text_model
    return text_model.encode([f"query: {text}"], normalize_embeddings=True)

def get_clip_text_embedding(text):
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

    if not mongo_manager.ready:
        mongo_manager.connect()
    print("[DEBUG] Mongo 연결 성공")

    db = mongo_manager.db
    product_collection = mongo_manager.products

    # 동의어 로드
    synonyms_doc = db["synonyms"].find_one({"_id": "korean"})
    if synonyms_doc is None or "dict" not in synonyms_doc:
        raise ValueError("동의어 사전을 찾을 수 없습니다.")
    synonyms = synonyms_doc["dict"]

    # Projection 필드 최적화
    projection = {
        "name": 1, "description": 1, "detail": 1,
        "imageEmbedding": 1, "textEmbedding": 1,
        "link": 1, "imageUrl": 1, "price": 1,
        "category": 1, "csv": 1
    }
    products = list(product_collection.find({}, projection))

    # 카테고리 필터링
    inferred = infer_category(query, db)
    print(f"[DEBUG] inferred 카테고리: {inferred}")
    if inferred:
        products = [p for p in products if inferred == (p.get("category") or "").strip()]
        print(f"[DEBUG] 카테고리 '{inferred}' 필터링 후 개수: {len(products)}")

    # 키워드 필터링
    keywords = query.split()
    keyword_filtered = [p for p in products if all(k in f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}" for k in keywords)]
    if len(keyword_filtered) >= 5:
        products = keyword_filtered
        print(f"[DEBUG] 키워드 필터 적용됨: {len(products)}개")

    items = products
    image_embeddings = np.array([p["imageEmbedding"] for p in products], dtype=np.float32)
    text_embeddings = np.array([p["textEmbedding"] for p in products], dtype=np.float32)

    queries = expand_query(query, synonyms)
    queries = queries[:3]
    print(f"[DEBUG] 동의어 확장 결과: {queries}")

    # 병렬 유사도 계산
    def compute_score(q):
        e5_embed = get_text_embedding(q)
        clip_embed = get_clip_text_embedding(q)
        sim_text = cosine_similarity(e5_embed, text_embeddings)[0]
        sim_image = cosine_similarity(clip_embed, image_embeddings)[0]
        return 0.6 * sim_text + 0.4 * sim_image

    with ThreadPoolExecutor() as executor:
        sim_results = list(executor.map(compute_score, queries))

    best_sim = max(sim_results, key=lambda s: max(s))
    best_indices = np.argsort(best_sim)[::-1]

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
            "유사도": float(best_sim[i]),
            "추천이유": f"쿼리 '{query}' 와 유사도 {float(best_sim[i]):.3f}"
        })

    return results
