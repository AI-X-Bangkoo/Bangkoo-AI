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

from utils.query_utils import (
    infer_category,
    expand_query,
    get_text_embedding,
    get_clip_text_embedding,
    compute_keyword_bonus,
    extract_color_from_caption,
    extract_keywords_from_query,
    extract_shape_from_caption
)
from utils.visual_color_utils import get_color_keywords_from_db

load_dotenv()

def hybrid_search(query, top_k=10):
    print("[DEBUG] hybrid_search 진입")

    # --- 모델 및 DB 연결 상태 확인 ---
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    if not mongo_manager.ready:
        mongo_manager.connect()
    print("[DEBUG] Mongo 연결 성공")

    db = mongo_manager.db
    product_collection = mongo_manager.products

    # --- 동의어 사전 로드 ---
    synonyms_doc = db["synonyms"].find_one({"_id": "korean"})
    if not synonyms_doc or "dict" not in synonyms_doc:
        raise ValueError("동의어 사전을 찾을 수 없습니다.")
    synonyms = synonyms_doc["dict"]

    # --- 카테고리 필터 조건 설정 ---
    query_filter = {}
    inferred = infer_category(query, db)
    print(f"[DEBUG] inferred 카테고리: {inferred}")
    if inferred:
        query_filter["category"] = inferred

    # --- MongoDB 텍스트 검색 조건 추가 ---
    query_filter["$text"] = {"$search": query}

    # --- 필요한 필드만 조회 ---
    projection = {
        "name": 1, "description": 1, "detail": 1,
        "imageEmbedding": 1, "textEmbedding": 1,
        "link": 1, "imageUrl": 1, "price": 1,
        "category": 1, "csv": 1
    }

    # --- MongoDB에서 검색 ---
    products = list(product_collection.find(query_filter, projection))
    print(f"[DEBUG] DB 조회 후 제품 개수: {len(products)}")
    if not products:
        return []

    # --- 색상 필터링 적용 ---
    color_key = extract_color_from_caption(query)
    if color_key:
        color_dict = get_color_keywords_from_db()
        color_synonyms = color_dict.get(color_key, [])
        filtered = []
        for p in products:
            text = f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}".lower()
            if any(s in text for s in color_synonyms):
                filtered.append(p)
        if filtered:
            print(f"[COLOR] '{color_key}' 관련 제품만 필터링: {len(filtered)}개")
            products = filtered

    # --- 형태 필터링 적용 ---
    shape_key, shape_synonyms = extract_shape_from_caption(query, db)
    if shape_key:
        filtered_shape = []
        for p in products:
            text = f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}".lower()
            if any(s in text for s in shape_synonyms):
                filtered_shape.append(p)
        if filtered_shape:
            print(f"[SHAPE] 형태 '{shape_key}' 관련 제품만 필터링: {len(filtered_shape)}개")
            products = filtered_shape
        else:
            print(f"[SHAPE] 형태 '{shape_key}' 관련 제품 없음 → 기존 결과 유지")

    # --- 임베딩 배열 구성 ---
    image_embeddings = np.array([p["imageEmbedding"] for p in products], dtype=np.float32)
    text_embeddings = np.array([p["textEmbedding"] for p in products], dtype=np.float32)

    # --- 쿼리 확장 (동의어 포함) ---
    queries = expand_query(query, synonyms)[:3]
    print(f"[DEBUG] 동의어 확장 결과: {queries}")

    # --- 텍스트 및 이미지 임베딩 기반 유사도 계산 --- 
    def compute_base_score(q):
        e5_embed = get_text_embedding(q)
        clip_embed = get_clip_text_embedding(q)
        sim_text = cosine_similarity(e5_embed, text_embeddings)[0]
        sim_image = cosine_similarity(clip_embed, image_embeddings)[0]
        return 0.6 * sim_text + 0.4 * sim_image

    with ThreadPoolExecutor() as executor:
        base_scores_list = list(executor.map(compute_base_score, queries))
    base_score = max(base_scores_list, key=lambda s: max(s))

    # --- 최종 점수 계산 (임베딩 점수 + 키워드 보너스) ---
    keywords = query.split()
    final_scores = []
    for idx, product in enumerate(products):
        bonus = compute_keyword_bonus(product, keywords)
        BONUS_SCALE = 0.1
        final_score = base_score[idx] + BONUS_SCALE * bonus
        final_scores.append(final_score)
    final_scores = np.array(final_scores)

    # --- 점수 기준 정렬 후 상위 top_k 결과 반환 ---
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