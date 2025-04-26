import os
import sys
import time
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from model_loader import model_manager
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import google.generativeai as genai
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
    extract_shape_from_caption,
    auto_insert_space  # fallback 용으로 남겨둠
)
from utils.visual_color_utils import get_color_keywords_from_db
from utils.markdown_utils import extract_json_from_markdown
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
okt = Okt()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.0-flash")

"""
최초 작성자: 김동규
최초 작성일: 2025-04-02
수정일: 2025-04-26 (김동규) (전처리/검색/후처리 및 전체 검색 파이프라인 흐름 주석 추가)

하이브리드 검색 시스템 (Hybrid Search Pipeline)

[전처리 단계]
1. 입력 쿼리 분석
    - 간단 패턴(색상+카테고리 조합) 감지 시 LLM 호출 생략
    - 일반 쿼리는 Gemini 2.0-flash를 호출하여 쿼리 교정, 속성(color, shape, category) 추출, 확장 쿼리 생성
2. 모델 준비 상태 확인
    - 모델 및 MongoDB 연결 상태 점검
3. 최종 속성 보완
    - 카테고리 자동 추론(infer_category)로 attributes 보완
    - 동의어 사전 로드(synonyms)

[검색 단계]
4. Atlas Search 기반 제품 후보 검색
    - refined_query를 기준으로 BM25 방식으로 색인 검색
    - 속성(color/shape) 기반 소프트 필터링 적용
5. BM25 및 벡터 유사도 계산 (병렬 처리)
    - BM25Okapi 기반 텍스트 토큰 유사도 점수 계산
    - textEmbedding + imageEmbedding 벡터 기반 cosine similarity 계산
6. LLM 속성 보너스 점수 계산
    - 추출된 속성 키워드 등장빈도 기반 보너스 점수 산출
7. 점수 가중합 (combine_scores)
    - 벡터 50%, BM25 30%, LLM 20% 가중합하여 최종 스코어 계산
8. 점수 임계치 필터링
    - 최종 점수가 0.5 이상인 후보만 선별
9. 필터링된 후보 정렬 및 추출
    - 결합 점수 기준 내림차순 정렬하여 top-k 추출

[후처리 단계]
10. 피드백 기반 Re-Ranking
    - 후보 제품의 CTR(click-through rate) 기반 점수 보정
    - 속성(strict filter) 일치 여부에 따라 추가 보정
    - 최종적으로 재정렬하여 반환
"""

# -------------------------------
# 동의어/매핑 사전
# -------------------------------
_color_doc    = mongo_manager.db["color_keywords"].find_one({"_id": "korean"}) or {}
_shape_doc    = mongo_manager.db["shape_keywords"].find_one({"_id": "korean"}) or {}
_category_doc = mongo_manager.db["category_keywords"].find_one({"_id": "korean"}) or {}
_stop_doc     = mongo_manager.db["stopwords_keywords"].find_one({"_id": "korean"}) or {}

_COLOR_SYNS    = _color_doc.get("dict", {})
_SHAPE_SYNS    = _shape_doc.get("dict", {})
_CATEGORY_SYNS = _category_doc.get("dict", {})
_STOPWORDS     = set(_stop_doc.get("words", []))

def get_color_synonyms():
    return _COLOR_SYNS

def get_shape_synonyms():
    return _SHAPE_SYNS

def get_category_synonyms():
    return _CATEGORY_SYNS

def get_stopwords():
    return _STOPWORDS

def tokenize_clean(text):
    tokens = okt.morphs(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]

_SIMPLE_Q_RE = re.compile(
    rf"^({'|'.join(get_color_synonyms().keys())})\s+({'|'.join(get_category_synonyms().keys())})$"
)

# ——————————————————————————————————————
# 전역 BM25 모델 캐시
# ——————————————————————————————————————

# 1) 모든 제품의 indexedTokens를 미리 불러와서 코퍼스 구성
_all_products = list(mongo_manager.products.find(
    {}, {"indexedTokens": 1, "link": 1}
))
# 2) 각 제품별 토큰 리스트
_global_corpus_tokens = []
#    link → corpus index 맵 (추후 subset 매핑용)
_global_link2idx = {}
for idx, doc in enumerate(_all_products):
    toks = doc.get("indexedTokens")
    if not toks:
        # indexedTokens 가 없으면 name/description/detail 에서 추출
        text = f"{doc.get('name','')} {doc.get('description','')} {doc.get('detail','')}".lower()
        toks = tokenize_clean(text)
    _global_corpus_tokens.append(toks)
    _global_link2idx[doc["link"]] = idx

# 3) BM25 모델 생성
_global_bm25 = BM25Okapi(_global_corpus_tokens)

# =============================================================================
# Gemini LLM API 호출 함수 (Gemini 2.0-flash 사용)
# =============================================================================
@lru_cache(maxsize=128)
def call_gemini_llm_all_in_one(query: str) -> dict:
    prompt = f"""
검색 쿼리 '{query}'에 대해 다음 3가지를 수행해 주세요:

1. **교정된 쿼리(corrected)**: 띄어쓰기 및 오탈자를 수정한 쿼리
2. **속성(attributes)**: 색상(color), 형태(shape), 카테고리(category) 추출
3. **확장된 표현(expanded)**: 의미가 유사한 대체 쿼리 표현 3개 이상

반드시 다음 형식의 **JSON**으로만 출력해 주세요:

{{
  "corrected": "...",
  "attributes": {{
    "color": "...",
    "shape": "...",
    "category": "..."
  }},
  "expanded": ["...", "...", "..."]
}}
"""
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    
    # 1) ```json … ``` 블록 또는 순수 JSON 배열만 꺼내기
    json_text = extract_json_from_markdown(raw_text)

    try:
        result = json.loads(json_text)
    except Exception as e:
        # print(f"[Gemini 통합 파싱 실패 → fallback 사용] {e}")
        # 기존 fallback 로직
        return {
            "corrected": query,
            "attributes": {
                "color": extract_color_from_caption(query),
                "shape": extract_shape_from_caption(query, mongo_manager.db)[0],
                "category": infer_category(query, mongo_manager.db)
            },
            "expanded": [query]
        }

    # 파싱에 성공했으면 기본값 채우기
    result.setdefault("corrected", query)
    result.setdefault("attributes", {"color": None, "shape": None, "category": None})
    result.setdefault("expanded", [query])
    return result

# =============================================================================
# 쿼리 전처리: Gemini 기반 쿼리 교정 및 속성 추출
# =============================================================================
def get_gemini_all_in_one(query: str):
    return call_gemini_llm_all_in_one(query)

# =============================================================================
# Atlas Search 및 제품 필터링
# =============================================================================
def atlas_search(refined_query, attributes, top_k=100):
    db = mongo_manager.db
    product_collection = mongo_manager.products
    
    synonyms_dict = db["synonyms"].find_one({"_id": "korean"})["dict"]
    expanded = expand_query(refined_query, synonyms_dict)
    
    must_clause = {
        "text": {"query": refined_query, "path": ["name","description","detail"]}
    }
    should_clauses = [
        {
            "text": {
                "query": kw,
                "path": ["name","description","detail"],
                "score": {"boost":{"value":2.0}}
            }
        }
        for kw in expanded
    ]

    should_category = []
    if attributes.get("category"):
        should_category = [{
            "term": {
                "query": attributes["category"],
                "path": "category",
                "score": {"boost":{"value":3.0}}
            }
        }]

    compound = {
        "must": [must_clause],
        "should": should_clauses + should_category
        # filter 제거
    }
    
    limit_count = top_k * 2

    pipeline = [
        {
            "$search": {
                "index":  "search_index",
                "compound": compound
            }
        },
        {"$limit": limit_count},
        {

            "$match": {
            "textEmbedding":   {"$exists": True, "$type": "array"},
            "imageEmbedding":  {"$exists": True, "$type": "array"},
            "$expr": {
            "$and": [
                {"$eq": [{"$size": "$textEmbedding"}, 768]},
                {"$eq": [{"$size": "$imageEmbedding"}, 1024]}
                ]
                }
            }
        },
        {
            "$project": {
                "_id":         0,
                "name":        1,
                "description": 1,
                "detail":      1,
                "textEmbedding":   1,
                "imageEmbedding":  1,
                "link":       1,
                "imageUrl":    1,
                "price":       1,
                "category":    1,
                "csv":         1,
                "model3dUrl":  1,
                "searchScore": {"$meta": "searchScore"}
            }
        }
    ]



    start = time.time()
    products = list(product_collection.aggregate(pipeline))
    end = time.time()
    # print(f"[Atlas Search 조회 소요 시간]: {end - start:.2f}초")
    # print(f"[DEBUG] DB 조회 후 제품 개수: {len(products)}")

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
            # print(f"[COLOR] '{color_key}' 관련 제품 필터링: {len(filtered)}개")
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
            # print(f"[SHAPE] 형태 '{shape_key}' 관련 제품 필터링: {len(filtered_shape)}개")
            products = filtered_shape
        else:
            print(f"[SHAPE] 형태 '{shape_key}' 관련 제품 없음 → 기존 결과 유지")

    return products

# =============================================================================
# BM25 기반 서치 (형태소 분석 및 BM25Okapi 사용)
# =============================================================================
def bm25_search(refined_query, products):
    # 1) 쿼리 토큰화
    query_tokens = tokenize_clean(refined_query)
    # 2) 전역 BM25로 전체 제품 점수 계산
    all_scores = np.array(_global_bm25.get_scores(query_tokens), dtype=float)

    # 3) 입력된 products 순서대로 점수 매핑
    scores = np.array([
        all_scores[_global_link2idx[p["link"]]]
        for p in products
    ], dtype=float)

    # 4) 0~1 정규화
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# =============================================================================
# 벡터 유사도 서치 (동적 확장 쿼리별 임베딩 기반 점수 계산)
# =============================================================================
def vector_similarity_search(expanded_queries, products):
    """
    CPU 전용 벡터 유사도 계산
    - expanded_queries: List[str]
    - products: List[dict], 각 dict에 'textEmbedding' (shape D_txt)과 'imageEmbedding' (shape D_img)이 numpy array로 들어있어야 함
    """
    # 1) 제품 임베딩 행렬 쌓기: text_matrix.shape = (N, D_txt), image_matrix.shape = (N, D_img)
    text_matrix  = np.stack([p["textEmbedding"]  for p in products], axis=0)
    image_matrix = np.stack([p["imageEmbedding"] for p in products], axis=0)

    # 2) 쿼리별 유사도 계산
    scores_list = []
    for q in expanded_queries:
        try:
            e5_vec   = get_text_embedding(q)    # (1, D_txt) 혹은 (D_txt,)
            clip_vec = get_clip_text_embedding(q)  # (1, D_img) 혹은 (D_img,)

            # 1차원 반환 시 2차원으로 바꿔주기
            if e5_vec.ndim == 1:
                e5_vec = e5_vec.reshape(1, -1)
            if clip_vec.ndim == 1:
                clip_vec = clip_vec.reshape(1, -1)

            # cosine_similarity → (1, N) 배열 반환
            sim_text  = cosine_similarity(e5_vec,  text_matrix).flatten()
            sim_image = cosine_similarity(clip_vec, image_matrix).flatten()

            # 가중합
            score = 0.6 * sim_text + 0.4 * sim_image
            scores_list.append(score)
        except Exception as e:
            print(f"[SKIP] '{q}' 임베딩 실패 → {e}")

    # 3) 쿼리별 스코어 중 최댓값 취하기
    if not scores_list:
        return None
    vector_scores = np.maximum.reduce(scores_list)
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
        return {"vector": 0.5, "bm25": 0.3, "llm": 0.2}

def adjust_scores_with_feedback(candidates, base_scores, feedback_weight):
    adjusted_scores = base_scores.copy()
    for i, candidate in enumerate(candidates):
        ctr = get_ctr(candidate)
        adjusted_scores[i] += feedback_weight * ctr
    return adjusted_scores

def re_rank_candidates_with_feedback(query, candidates, base_scores, attributes):
    start = time.time()
    # 1) A/B 가중치는 그대로
    variant_weights = get_reranking_weights()
    # print(f"[A/B Variant] 적용된 가중치: {variant_weights}")

    # 2) 후보 ID 리스트 수집
    candidate_ids = [cand["링크"] for cand in candidates]

    # 3) 한 번에 feedback 문서들 가져오기
    feedback_docs = list(
        mongo_manager.db["candidate_feedback"]
        .find({"candidate_id": {"$in": candidate_ids}}, 
              {"candidate_id": 1, "clicks": 1, "impressions": 1})
    )
    # 4) candidate_id → feedback 매핑
    feedback_map = {
        doc["candidate_id"]: doc for doc in feedback_docs
    }

    # 5) batch-CTR 보정 함수로 교체
    def adjust_scores_with_feedback_batch(cands, scores, weight, fb_map):
        adjusted = scores.copy()
        for i, cand in enumerate(cands):
            fb = fb_map.get(cand["링크"])
            if fb and fb.get("impressions", 0) > 0:
                ctr = fb.get("clicks", 0) / fb["impressions"]
            else:
                ctr = 0.0
            adjusted[i] += weight * ctr
        return adjusted

    # 6) feedback 보정 (이제 DB 왕복은 한 번만)
    fb_weight = variant_weights.get("feedback", 0)
    adjusted_scores = adjust_scores_with_feedback_batch(
        candidates, base_scores, fb_weight, feedback_map
    )

    # 7) 속성(strict filter) 보정은 그대로
    adjusted_scores = adjust_scores_with_strict_filter(
        candidates, adjusted_scores, attributes, penalty=0.2
    )

    # 8) 최종 정렬
    indices = np.argsort(adjusted_scores)[::-1]
    re_ranked = [candidates[i] for i in indices]

    end = time.time()
    # print(f"피드백 보정 및 속성 보정 적용 소요 시간: {end - start:.2f}초")
    return re_ranked


# =============================================================================
# 하이브리드 서치 + Re-Ranking 최종 함수 (피드백, A/B 테스트, 속성 보정 적용)
# =============================================================================
def hybrid_search(query, top_k=None):
    # print("[DEBUG] hybrid_search 진입")
    start1 = time.time()

    # 1) simple-pattern 체크: "색상 카테고리" 형태 쿼리면 LLM 스킵
    m = _SIMPLE_Q_RE.match(query)
    if m:
        color, category = m.groups()
        refined_query    = query
        attributes       = {"color": color, "shape": None, "category": category}
        expanded_queries = [query]
        # print(f"[DEBUG] 룰 기반 속성 추출 → refined_query={refined_query}, attributes={attributes}")
        gemini_time = 0.0
    else:
        # 2) 그 외에만 LLM 호출
        start = time.time()
        gemini_result    = get_gemini_all_in_one(query)
        refined_query    = gemini_result["corrected"]
        attributes       = gemini_result["attributes"]
        expanded_queries = gemini_result["expanded"]
        gemini_time = time.time() - start
    #     print(f"[LLM 처리 소요 시간]: {gemini_time:.2f}초")
    # print(f"[DEBUG] 정제된 쿼리: {refined_query}, 추출된 속성: {attributes}")
    
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    if not mongo_manager.ready:
        mongo_manager.connect()
    # print("[DEBUG] Mongo 연결 성공")
    
    db = mongo_manager.db
    synonyms_doc = db["synonyms"].find_one({"_id": "korean"})
    if not synonyms_doc or "dict" not in synonyms_doc:
        raise ValueError("동의어 사전을 찾을 수 없습니다.")
    synonyms = synonyms_doc["dict"]
    
    inferred = infer_category(refined_query, db)
    if inferred and not attributes.get("category"):
        attributes["category"] = inferred
    # print(f"[DEBUG] 최종 카테고리: {attributes.get('category')}")
    
    products = atlas_search(refined_query, attributes, top_k=top_k or 100)
    if not products:
        # print("[ERROR] 제품 조회 결과 없음")
        return []
    
    # print(f"[DEBUG] 확장된 쿼리: {expanded_queries}")
    
     # 2) BM25와 벡터 유사도를 병렬로 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bm25  = executor.submit(bm25_search, refined_query, products)
        future_vec   = executor.submit(vector_similarity_search, expanded_queries, products)

        # 두 작업이 다 끝날 때까지 기다림
        bm25_scores   = future_bm25.result()
        vector_scores = future_vec.result()

    # 실패 처리
    if vector_scores is None:
        # print("[ERROR] 모든 쿼리에 대해 임베딩 실패 → 빈 결과 반환")
        return []
    if bm25_scores is None:
        # BM25가 실패해도 0점 배열로 대체
        bm25_scores = np.zeros_like(vector_scores)

    # print(f"[DEBUG] BM25 점수:   {bm25_scores}")
    # print(f"[DEBUG] 벡터 점수: {vector_scores}")
    
    start = time.time()
    llm_bonus = compute_llm_bonus(products, attributes)
    end = time.time()
    # print(f"[Atlas LLM 보너스 점수 소요 시간]: {end - start:.2f}초")
    # print(f"[DEBUG] LLM 보너스 점수: {llm_bonus}")
    
    # 10. 기본 점수 결합
    weights = {"vector": 0.5, "bm25": 0.3, "llm": 0.2}
    start = time.time()
    final_scores = combine_scores(vector_scores, bm25_scores, llm_bonus, weights)
    end = time.time()
    # print(f"[Atlas 최종 결합 점수 소요 시간]: {end - start:.2f}초")
    # print(f"[DEBUG] 최종 결합 점수: {final_scores}")

    # 11. 임계치(threshold) 미만인 후보 제외하기
    THRESHOLD = 0.5  # 예시: 최종 결합 점수가 0.5 미만인 후보는 제거 (정규화된 스코어 기준)
    high_score_indices = np.where(final_scores >= THRESHOLD)[0]
    if high_score_indices.size == 0:
        # print("[DEBUG] 모든 후보의 점수가 낮습니다. 임계치를 낮춰보세요.")
        high_score_indices = np.arange(len(final_scores))
    start = time.time()
    filtered_final_scores = final_scores[high_score_indices]
    end = time.time()
    # print(f"[Atlas 임계치 미만 후보 제외 소요 시간]: {end - start:.2f}초")
    # print(f"[DEBUG] 임계치 {THRESHOLD} 이상인 후보 인덱스: {high_score_indices}")

    start = time.time()
    # high_score_indices를 사용해 원래 제품 리스트에서 후보들(filtered_products) 추출
    filtered_products = [products[i] for i in high_score_indices]
    # 필터링된 후보들의 점수(filtered_final_scores) 기준 내림차순 정렬 (인덱스 재정렬)
    sorted_filtered_indices = np.argsort(filtered_final_scores)[::-1]
    end = time.time()
    # print(f"[후보 추출 및 정렬 소요 시간]: {end - start:.2f}초")
    
    # 12. 상위 후보 추출 (인상 로그 로직 제거)
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
            "추천이유": f"쿼리 '{refined_query}' 와 결합 유사도 {float(filtered_final_scores[idx]):.3f}",
            # 범석 추가
            "glb이미지": item.get("model3dUrl",None)
        }
        candidates.append(candidate)

    # 13. (이제 로그는 Java 백엔드가 담당)
    base_scores = np.array([cand["유사도"] for cand in candidates])
    re_ranked_candidates = re_rank_candidates_with_feedback(
        refined_query, candidates, base_scores, attributes
    )

    end1 = time.time()
    print(f"[검색 총 소요 시간]: {end1 - start1:.2f}초")
    return re_ranked_candidates
