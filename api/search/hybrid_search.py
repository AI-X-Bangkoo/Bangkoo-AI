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

def compute_keyword_bonus(product, keywords):
    """
    제품의 name, description, detail 필드에서 쿼리 키워드가 얼마나 많이 매칭되는지를 평가하려고 생성
    반환 값은 0과 1 사이의 값으로, 1에 가까울수록 모든 키워드가 포함되어 있다는 의미
    """
    text = f"{product.get('name','')} {product.get('description','')} {product.get('detail','')}"
    matched = sum(1 for k in keywords if k in text)
    return matched / len(keywords) if keywords else 0

def hybrid_search(query, top_k=10):
    print("[DEBUG] hybrid_search 진입")

    # 모델과 MongoDB 연결 상태 확인
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    if not mongo_manager.ready:
        mongo_manager.connect()
    print("[DEBUG] Mongo 연결 성공")

    db = mongo_manager.db
    product_collection = mongo_manager.products

    # 동의어 사전 로드
    synonyms_doc = db["synonyms"].find_one({"_id": "korean"})
    if not synonyms_doc or "dict" not in synonyms_doc:
        raise ValueError("동의어 사전을 찾을 수 없습니다.")
    synonyms = synonyms_doc["dict"]

    # MongoDB 쿼리 조건 빌드:
    # 1. 카테고리 필터링
    query_filter = {}
    inferred = infer_category(query, db)
    print(f"[DEBUG] inferred 카테고리: {inferred}")
    if inferred:
        query_filter["category"] = inferred

    # 2. 텍스트 인덱스를 활용한 키워드 필터링
    query_filter["$text"] = {"$search": query}

    # Projection 필드 최적화
    projection = {
        "name": 1, "description": 1, "detail": 1,
        "imageEmbedding": 1, "textEmbedding": 1,
        "link": 1, "imageUrl": 1, "price": 1,
        "category": 1, "csv": 1
    }

    # DB에서 조건에 맞는 제품 조회
    products = list(product_collection.find(query_filter, projection))
    print(f"[DEBUG] DB 조회 후 제품 개수: {len(products)}")
    if not products:
        return []

    # 임베딩 계산을 위한 배열 생성
    image_embeddings = np.array([p["imageEmbedding"] for p in products], dtype=np.float32)
    text_embeddings = np.array([p["textEmbedding"] for p in products], dtype=np.float32)

    # 동의어 확장: 쿼리 확장 후 상위 3개 후보 사용
    queries = expand_query(query, synonyms)[:3]
    print(f"[DEBUG] 동의어 확장 결과: {queries}")

    # 임베딩 기반 유사도 계산 (Base Score)
    def compute_base_score(q):
        e5_embed = get_text_embedding(q)
        clip_embed = get_clip_text_embedding(q)
        sim_text = cosine_similarity(e5_embed, text_embeddings)[0]
        sim_image = cosine_similarity(clip_embed, image_embeddings)[0]
        return 0.6 * sim_text + 0.4 * sim_image

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        base_scores_list = list(executor.map(compute_base_score, queries))
    base_score = max(base_scores_list, key=lambda s: max(s))  # 여기서 선택 기준은 최댓값임

    # 키워드 보너스 계산을 위해 쿼리를 단어 단위로 분리
    keywords = query.split()

    # 각 제품에 대해 임베딩 기반 점수와 키워드 보너스를 계산하여 최종 점수 산출
    final_scores = []
    for idx, product in enumerate(products):
        bonus = compute_keyword_bonus(product, keywords)  # 이거 0~1 사이의 값을 반환
        # BONUS_SCALE은 보너스의 반영 강도를 조절 (0.1은 최대 0.1만큼 점수를 추가함)
        BONUS_SCALE = 0.1
        final_score = base_score[idx] + BONUS_SCALE * bonus
        final_scores.append(final_score)
    final_scores = np.array(final_scores)

    # 최종 점수에 따라 제품 재정렬
    best_indices = np.argsort(final_scores)[::-1]

    results = []
    for i in best_indices[:top_k]:
        item = products[i]
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
            "유사도": float(final_scores[i]),
            "추천이유": f"쿼리 '{query}' 와 결합 유사도 {float(final_scores[i]):.3f}"
        })

    return results


