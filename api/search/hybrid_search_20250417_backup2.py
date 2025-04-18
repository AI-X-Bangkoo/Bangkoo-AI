import os
import sys
import time
import json
import re
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from model_loader import model_manager
import torch
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mongo_manager import mongo_manager
from utils.query_utils import (
    infer_category,
    expand_query,  # 기존 동의어 사전 활용 및 동적 LLM 확장과 병행 가능함
    get_text_embedding,
    get_clip_text_embedding,
    compute_keyword_bonus,
    extract_color_from_caption,
    extract_keywords_from_query,
    extract_shape_from_caption,
    auto_insert_space  # fallback 용으로 남겨둠
)
from utils.visual_color_utils import get_color_keywords_from_db

from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
okt = Okt()

# -------------------------------
# 동의어/매핑 사전
# -------------------------------
def get_color_synonyms():
    db = mongo_manager.db
    doc = db["color_keywords"].find_one({"_id": "korean"})
    return doc["dict"]

def get_shape_synonyms():
    db = mongo_manager.db
    doc = db["shape_keywords"].find_one({"_id": "korean"})
    return doc["dict"]

def get_category_synonyms():
    db = mongo_manager.db
    doc = db["category_keywords"].find_one({"_id": "korean"})
    return doc["dict"]

# =============================================================================
# Gemini LLM API 호출 함수 (Gemini 1.5-flash 사용)
# =============================================================================
def call_gemini_llm_query_correction(query):    
    prompt = f"상품 검색 쿼리의 띄어쓰기와 오탈자를 교정해 주세요. 입력: '{query}'"
    response = model.generate_content(prompt)
    corrected = response.text.strip()
    
    return corrected

def call_gemini_llm_attribute_extraction(query):
    prompt = (
        f"다음 상품 검색 쿼리에서 제품의 속성을 추출해 주세요. 속성은 색상(color), 형태(shape), 카테고리(category)로 구분하며, "
        f"쿼리가 단일 키워드라면 해당 키워드를 카테고리로 간주하여 반환해 주세요.\n"
        f"쿼리: '{query}'\n출력 예: {{\"color\": null, \"shape\": null, \"category\": \"조명\"}}"
    )
    response = model.generate_content(prompt)
    try:
        attributes = json.loads(response.text)
    except Exception as e:
        print(f"[LLM ATTRIBUTE 추출 오류] {e}")
        inferred_cat = infer_category(query, mongo_manager.db)
        inferred_shape, _ = extract_shape_from_caption(query, mongo_manager.db)
        inferred_color = extract_color_from_caption(query)
        attributes = {"color": inferred_color, "shape": inferred_shape, "category": inferred_cat}
    return attributes

def call_gemini_llm_query_expansion(query):
    prompt = (
        f"상품 검색 쿼리 '{query}'에 대해 동의어나 유사한 대체 표현을 3개 이상 생성해 주세요. "
        f"출력은 오직 JSON 배열 형식만 제공해 주세요. 예를 들어, [\"동그란 테이블\", \"원형 테이블\", \"라운드 테이블\"]와 같이 출력해 주세요."
    )
    response = model.generate_content(prompt)
    # 응답 텍스트를 정제: 코드 블록 마크다운 제거
    resp_text = response.text.strip()
    # 만약 문자열이 "```" 로 시작하면 제거
    if resp_text.startswith("```"):
        # 첫 번째 줄에 ```json (또는 ``` 로 시작하는 경우)를 제거
        lines = resp_text.splitlines()
        # 제거: 첫 줄(마크다운 시작)과 마지막 줄(마크다운 종료)
        if len(lines) >= 2:
            resp_text = "\n".join(lines[1:-1]).strip()
        else:
            resp_text = ""
    try:
        synonyms = json.loads(resp_text)
        if not isinstance(synonyms, list):
            synonyms = [query]
    except Exception as e:
        print(f"[LLM 확장 오류] {e}")
        synonyms = [query]
        
    return synonyms


# =============================================================================
# 쿼리 전처리: Gemini 기반 쿼리 교정 및 속성 추출
# =============================================================================
def refined_query_processing(query):
    corrected_query = call_gemini_llm_query_correction(query)
    attributes = call_gemini_llm_attribute_extraction(corrected_query)
    return corrected_query, attributes

def expand_query_using_llm(query):
    return call_gemini_llm_query_expansion(query)

# =============================================================================
# Atlas Search 및 제품 필터링
# =============================================================================
def atlas_search(refined_query, attributes):
    db = mongo_manager.db
    product_collection = mongo_manager.products

    query_filter = {}
    if attributes.get("category"):
        query_filter["category"] = attributes["category"]

    # pipeline = [
    #     {
    #         "$search": {
    #             "index": "search_index",
    #             "text": {
    #                 "query": refined_query,
    #                 "path": ["name", "description", "detail"]
    #             }
    #         }
    #     },
    #     {
    #         "$project": {
    #             "_id": 0,
    #             "name": 1,
    #             "description": 1,
    #             "detail": 1,
    #             "imageEmbedding": 1,
    #             "textEmbedding": 1,
    #             "link": 1,
    #             "imageUrl": 1,
    #             "price": 1,
    #             "category": 1,
    #             "csv": 1,
    #             "searchScore": {"$meta": "searchScore"}
    #         }
    #     },
    #     {"$limit": 200}
    # ]
    pipeline = [
        {
            "$search": {
                "index": "search_index",
                "compound": {
                    "must": [
                        {
                            "text": {
                                "query": refined_query,
                                "path": ["name", "description", "detail"]
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "query": attributes["category"],
                                "path": "category"
                            }
                        }
                    ]
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "description": 1,
                "detail": 1,
                "imageEmbedding": 1,
                "textEmbedding": 1,
                "link": 1,
                "imageUrl": 1,
                "price": 1,
                "category": 1,
                "csv": 1,
                "searchScore": {"$meta": "searchScore"}
            }
        },
        {"$limit": 200}
    ]


    start = time.time()
    products = list(product_collection.aggregate(pipeline))
    end = time.time()
    print(f"[Atlas Search 조회 소요 시간]: {end - start:.2f}초")
    print(f"[DEBUG] DB 조회 후 제품 개수: {len(products)}")

    # 색상 필터링
    # DB에서 색상 동의어 사전을 가져옴
    color_keywords = get_color_synonyms()
    color_key = attributes.get("color") if attributes.get("color") else extract_color_from_caption(refined_query)
    if color_key:
        # DB에 저장된 색상 동의어 배열을 사용
        color_synonyms = color_keywords.get(color_key.lower(), [color_key.lower()])
        filtered = []
        for p in products:
            text = f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}".lower()
            if any(s in text for s in color_synonyms):
                filtered.append(p)
        if filtered:
            print(f"[COLOR] '{color_key}' 관련 제품 필터링: {len(filtered)}개")
            products = filtered
    
    # DB 동의어 사용
    shape_keywords = get_shape_synonyms()
    if attributes.get("shape"):
        shape_key = attributes.get("shape")
        shape_synonyms = shape_keywords.get(shape_key.lower(), [shape_key.lower()])
    else:
        shape_key, _ = extract_shape_from_caption(refined_query, db)
        shape_synonyms = shape_keywords.get(shape_key.lower(), [shape_key.lower()]) if shape_key else []
    if shape_key and shape_synonyms:
        filtered_shape = []
        for p in products:
            text = f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}".lower()
            if any(s in text for s in shape_synonyms):
                filtered_shape.append(p)
        if filtered_shape:
            print(f"[SHAPE] 형태 '{shape_key}' 관련 제품 필터링: {len(filtered_shape)}개")
            products = filtered_shape
        else:
            print(f"[SHAPE] 형태 '{shape_key}' 관련 제품 없음 → 기존 결과 유지")

    return products

# =============================================================================
# BM25 기반 서치 (형태소 분석 및 BM25Okapi 사용)
# =============================================================================
def bm25_search(refined_query, products):
    corpus = []
    # 제품 텍스트 캐싱: 한 번 소문자로 변환한 텍스트를 저장
    product_texts = []
    for product in products:
        text = f"{product.get('name', '')} {product.get('description', '')} {product.get('detail', '')}".lower()
        product_texts.append(text)
        tokens = okt.morphs(text)
        corpus.append(tokens)
    query_tokens = okt.morphs(refined_query.lower())
    bm25 = BM25Okapi(corpus)
    scores = np.array(bm25.get_scores(query_tokens))
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# =============================================================================
# 벡터 유사도 서치 (동적 확장 쿼리별 임베딩 기반 점수 계산)
# =============================================================================
def vector_similarity_search(expanded_queries, products):
    # 이미 배열 형태로 만들어둔 임베딩 사용
    image_embeddings = np.array([p["imageEmbedding"] for p in products], dtype=np.float32)
    text_embeddings = np.array([p["textEmbedding"] for p in products], dtype=np.float32)
    
    # 필요 시 PyTorch 텐서로 전환 및 GPU 배치 (예시, 실제 GPU 사용 환경에 따라 조정)
    img_emb = torch.tensor(image_embeddings)  # .to('cuda') 로 GPU 할당 가능
    txt_emb = torch.tensor(text_embeddings)
    
    query_scores_list = []
    for q in expanded_queries:
        try:
            e5_embed = get_text_embedding(q)  # 이미 np.array 반환한다고 가정
            clip_embed = get_clip_text_embedding(q)
            e5_embed = torch.tensor(e5_embed)  # .to('cuda')
            clip_embed = torch.tensor(clip_embed)
            # Cosine similarity 계산 시, 텐서를 사용하면 GPU 가속 가능
            sim_text = torch.nn.functional.cosine_similarity(e5_embed, txt_emb).cpu().numpy()
            sim_image = torch.nn.functional.cosine_similarity(clip_embed, img_emb).cpu().numpy()
            score = 0.6 * sim_text + 0.4 * sim_image
            query_scores_list.append(score)
        except Exception as e:
            print(f"[SKIP] '{q}' 임베딩 실패 → {e}")
    if not query_scores_list:
        return None
    vector_scores = np.maximum.reduce(query_scores_list)
    return vector_scores


# =============================================================================
# LLM 보너스: 속성 값 등장빈도 기반 보너스 점수 산출
# =============================================================================
def compute_llm_bonus(products, attributes):
    bonus = np.zeros(len(products))
    bonus_keywords = [v for k, v in attributes.items() if v]
    
    # 미리 제품 텍스트를 캐싱 (소문자 처리)
    product_texts = [f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}".lower() for p in products]
    
    for idx, text in enumerate(product_texts):
        bonus[idx] = sum(text.count(keyword.lower()) for keyword in bonus_keywords)
    if bonus.max() > 0:
        bonus = bonus / bonus.max()
    return bonus


# =============================================================================
# 점수 결합: 벡터, BM25, LLM 보너스 점수를 가중합하여 최종 스코어 산출
# =============================================================================
def combine_scores(vector_scores, bm25_scores, llm_bonus, weights):
    final_scores = weights["vector"] * vector_scores
    final_scores += weights["bm25"] * bm25_scores
    final_scores += weights["llm"] * llm_bonus
    return final_scores

# =============================================================================
# 엄격한 속성 필터링 및 보정: 속성 불일치 시 페널티 부여 (동의어 사전 활용)
# =============================================================================
def adjust_scores_with_strict_filter(candidates, base_scores, attributes, penalty=0.2):
    adjusted_scores = base_scores.copy()
    category_synonyms = get_category_synonyms()
    color_synonyms = get_color_synonyms()
    shape_synonyms = get_shape_synonyms()

    for i, cand in enumerate(candidates):
        candidate_text = f"{cand.get('name', '')} {cand.get('description', '')} {cand.get('detail', '')}".lower()
        # 카테고리 검사
        if attributes.get("category"):
            cat = attributes["category"].lower()
            cand_category = cand.get("category", "").lower()
            cat_syns = category_synonyms.get(cat, [cat])
            if not any(s in cand_category or s in candidate_text for s in cat_syns):
                adjusted_scores[i] -= penalty
        # 색상 검사
        if attributes.get("color"):
            color = attributes["color"].lower()
            col_syns = color_synonyms.get(color, [color])
            if not any(s in candidate_text for s in col_syns):
                adjusted_scores[i] -= penalty
        # 형태 검사
        if attributes.get("shape"):
            shape = attributes["shape"].lower()
            sh_syns = shape_synonyms.get(shape, [shape])
            if not any(s in candidate_text for s in sh_syns):
                adjusted_scores[i] -= penalty
    return adjusted_scores


# =============================================================================
# 사용자 피드백 기반 로그 및 재정렬 (A/B 테스트 포함)
# =============================================================================
def log_impression(candidate):
    db = mongo_manager.db
    feedback_col = db["candidate_feedback"]
    candidate_id = candidate.get("링크", candidate.get("이름"))
    feedback = feedback_col.find_one({"candidate_id": candidate_id})
    if feedback:
        feedback_col.update_one({"candidate_id": candidate_id}, {"$inc": {"impressions": 1}})
    else:
        feedback_col.insert_one({
            "candidate_id": candidate_id,
            "clicks": 0,
            "impressions": 1,
            "last_updated": time.time()
        })
    
def log_click(candidate):
    db = mongo_manager.db
    feedback_col = db["candidate_feedback"]
    candidate_id = candidate.get("링크", candidate.get("이름"))
    feedback = feedback_col.find_one({"candidate_id": candidate_id})
    if feedback:
        feedback_col.update_one({"candidate_id": candidate_id}, {"$inc": {"clicks": 1}})
    else:
        feedback_col.insert_one({
            "candidate_id": candidate_id,
            "clicks": 1,
            "impressions": 0,
            "last_updated": time.time()
        })

def get_ctr(candidate):
    db = mongo_manager.db
    feedback_col = db["candidate_feedback"]
    candidate_id = candidate.get("링크", candidate.get("이름"))
    feedback = feedback_col.find_one({"candidate_id": candidate_id})
    if feedback and feedback.get("impressions", 0) > 0:
        return feedback.get("clicks", 0) / feedback.get("impressions", 0)
    return 0.0

def get_reranking_weights():
    db = mongo_manager.db
    config_col = db["search_config"]
    config = config_col.find_one({"config": "reranking_weights"})
    if config:
        config.pop("_id", None)
        config.pop("config", None)
        return config
    else:
        return {"vector": 0.5, "bm25": 0.2, "llm": 0.2, "feedback": 0.1}

def adjust_scores_with_feedback(candidates, base_scores, feedback_weight):
    adjusted_scores = base_scores.copy()
    for i, candidate in enumerate(candidates):
        ctr = get_ctr(candidate)
        adjusted_scores[i] += feedback_weight * ctr
    return adjusted_scores

def re_rank_candidates_with_feedback(query, candidates, base_scores, attributes):
    variant_weights = get_reranking_weights()
    print(f"[A/B Variant] 적용된 가중치: {variant_weights}")
    
    # 피드백 보정
    adjusted_scores = adjust_scores_with_feedback(candidates, base_scores, feedback_weight=variant_weights.get("feedback", 0))
    # 추가적으로 엄격한 속성 보정 적용
    adjusted_scores = adjust_scores_with_strict_filter(candidates, adjusted_scores, attributes, penalty=0.2)
    
    indices = np.argsort(adjusted_scores)[::-1]
    re_ranked = [candidates[i] for i in indices]
    return re_ranked

# =============================================================================
# 하이브리드 서치 + Re-Ranking 최종 함수 (피드백, A/B 테스트, 엄격한 속성 보정 적용)
# =============================================================================
def hybrid_search(query, top_k=None):
    print("[DEBUG] hybrid_search 진입")
    start1 = time.time()
    start = time.time()
    refined_query, attributes = refined_query_processing(query)
    end = time.time()
    print(f"[사용자 피드백 기반 로그 및 재정렬 소요 시간]: {end - start:.2f}초")
    print(f"[DEBUG] 정제된 쿼리: {refined_query}, 추출된 속성: {attributes}")
    
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    if not mongo_manager.ready:
        mongo_manager.connect()
    print("[DEBUG] Mongo 연결 성공")
    
    db = mongo_manager.db
    synonyms_doc = db["synonyms"].find_one({"_id": "korean"})
    if not synonyms_doc or "dict" not in synonyms_doc:
        raise ValueError("동의어 사전을 찾을 수 없습니다.")
    synonyms = synonyms_doc["dict"]
    
    inferred = infer_category(refined_query, db)
    if inferred and not attributes.get("category"):
        attributes["category"] = inferred
    print(f"[DEBUG] 최종 카테고리: {attributes.get('category')}")
    
    products = atlas_search(refined_query, attributes)
    if not products:
        print("[ERROR] 제품 조회 결과 없음")
        return []
    
    expanded_queries = expand_query_using_llm(refined_query)
    print(f"[DEBUG] 확장된 쿼리: {expanded_queries}")
    
    start = time.time()
    bm25_scores = bm25_search(refined_query, products)
    end = time.time()
    print(f"[Atlas BM25 점수 소요 시간]: {end - start:.2f}초")
    print(f"[DEBUG] BM25 점수: {bm25_scores}")
    
    start = time.time()
    vector_scores = vector_similarity_search(expanded_queries, products)
    end = time.time()
    print(f"[Atlas 벡터 유사도 점수 소요 시간]: {end - start:.2f}초")
    if vector_scores is None:
        print("[ERROR] 모든 쿼리에 대해 임베딩 실패 → 빈 결과 반환")
        return []
    print(f"[DEBUG] 벡터 유사도 점수: {vector_scores}")
    
    start = time.time()
    llm_bonus = compute_llm_bonus(products, attributes)
    end = time.time()
    print(f"[Atlas LLM 보너스 점수 소요 시간]: {end - start:.2f}초")
    print(f"[DEBUG] LLM 보너스 점수: {llm_bonus}")
    
    # 10. 기본 점수 결합
    weights = {"vector": 0.5, "bm25": 0.3, "llm": 0.2}
    start = time.time()
    final_scores = combine_scores(vector_scores, bm25_scores, llm_bonus, weights)
    end = time.time()
    print(f"[Atlas 최종 결합 점수 소요 시간]: {end - start:.2f}초")
    print(f"[DEBUG] 최종 결합 점수: {final_scores}")

    # 11. 임계치(threshold) 미만인 후보 제외하기
    THRESHOLD = 0.5  # 예시: 최종 결합 점수가 0.5 미만인 후보는 제거 (정규화된 스코어 기준)
    high_score_indices = np.where(final_scores >= THRESHOLD)[0]
    if high_score_indices.size == 0:
        print("[DEBUG] 모든 후보의 점수가 낮습니다. 임계치를 낮춰보세요.")
        high_score_indices = np.arange(len(final_scores))
    filtered_final_scores = final_scores[high_score_indices]
    print(f"[DEBUG] 임계치 {THRESHOLD} 이상인 후보 인덱스: {high_score_indices}")

    # high_score_indices를 사용해 원래 제품 리스트에서 후보들(filtered_products) 추출
    filtered_products = [products[i] for i in high_score_indices]
    # 필터링된 후보들의 점수(filtered_final_scores) 기준 내림차순 정렬 (인덱스 재정렬)
    sorted_filtered_indices = np.argsort(filtered_final_scores)[::-1]

    # 12. 상위 후보 추출 및 인상 로그 기록 (임계치 적용 후)
    candidates = []
    limit = top_k if top_k is not None else len(filtered_products)
    for idx in sorted_filtered_indices[:limit]:
        item = filtered_products[idx]
        candidate = {
            "이름": item["name"],
            "설명": item["description"],
            "상세설명": item.get("detail", ""),
            "링크": item["link"],
            "이미지": item["imageUrl"],
            "할인가": item.get("price", "-"),
            "정상가": item.get("price", "-"),
            "카테고리": item.get("category"),
            "csv": item.get("csv", ""),
            "유사도": float(filtered_final_scores[idx]),
            "추천이유": f"쿼리 '{refined_query}' 와 결합 유사도 {float(filtered_final_scores[idx]):.3f}"
        }
        log_impression(candidate)
        candidates.append(candidate)


    base_scores = np.array([cand["유사도"] for cand in candidates])
    re_ranked_candidates = re_rank_candidates_with_feedback(refined_query, candidates, base_scores, attributes)
    
    end1 = time.time()
    print(f"[검색 총 소요 시간]: {end1 - start1:.2f}초")
    return re_ranked_candidates
