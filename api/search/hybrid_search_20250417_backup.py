import os
import sys
import time
import json
import random
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

import google.generativeai as genai
from rank_bm25 import BM25Okapi
from mongo_manager import mongo_manager
from model_loader import model_manager
from utils.query_utils import (
    infer_category,
    get_text_embedding,
    get_clip_text_embedding,
    extract_color_from_caption,
    extract_shape_from_caption,
    auto_insert_space
)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# 동의어/매핑 사전
# -------------------------------
def get_synonym_dict(collection_name):
    return mongo_manager.db[collection_name].find_one({"_id": "korean"}).get("dict", {})

# =============================================================================
# 벡터 유사도 (CPU numpy)
# =============================================================================
def vector_similarity_search(products, expanded_queries):
    start = time.time()
    # DB에서 가져온 E5(textEmbedding) 및 CLIP(imageEmbedding) 배열
    text_embs = np.array([p['textEmbedding'] for p in products], dtype=np.float32)
    img_embs = np.array([p['imageEmbedding'] for p in products], dtype=np.float32)
    scores_list = []
    for q in expanded_queries:
        try:
            e5 = get_text_embedding(q)  # shape (1, D_e5)
            clip = get_clip_text_embedding(q)  # shape (1, D_clip)
            sim_text = cosine_similarity(e5, text_embs)[0]
            sim_img = cosine_similarity(clip, img_embs)[0]
            scores_list.append(0.6 * sim_text + 0.4 * sim_img)
        except Exception as e:
            print(f"[SKIP] '{q}' 임베딩 실패 → {e}")
    if not scores_list:
        return np.zeros(len(products), dtype=float)
    vector_scores = np.max(np.stack(scores_list, axis=0), axis=0)
    print(f"[벡터 유사도 계산 소요 시간]: {time.time()-start:.2f}s")
    return vector_scores

# =============================================================================
def call_gemini_llm_query_correction(query):
    prompt = f"상품 검색 쿼리의 띄어쓰기와 오탈자를 교정해 주세요. 입력: '{query}'"
    return model.generate_content(prompt).text.strip()

def call_gemini_llm_attribute_extraction(query):
    prompt = (
        f"다음 상품 검색 쿼리에서 속성(color, shape, category)을 JSON으로 추출하세요."
        f" 단일 키워드라면 category로 사용하세요. 쿼리: '{query}'\n"
        f"예: {{\"color\": null, \"shape\": null, \"category\": \"조명\"}}"
    )
    resp = model.generate_content(prompt).text.strip()
    try:
        return json.loads(resp)
    except:
        cat = infer_category(query, mongo_manager.db)
        shape, _ = extract_shape_from_caption(query, mongo_manager.db)
        color = extract_color_from_caption(query)
        return {"color": color, "shape": shape, "category": cat}

# =============================================================================
# 쿼리 전처리
# =============================================================================
def refined_query_processing(query):
    corrected = call_gemini_llm_query_correction(query)
    attrs = call_gemini_llm_attribute_extraction(corrected)
    return corrected, attrs

# =============================================================================
# Atlas Search + 필터링
# =============================================================================
def atlas_search(refined_query, attrs):
    pipeline = [
        {"$search": {"index": "search_index", "text": {"query": refined_query, "path": ["name","description","detail"]}}},
        {"$project": {"_id":0, "name":1, "description":1, "detail":1,
                       "imageEmbedding":1, "textEmbedding":1, "link":1,
                       "imageUrl":1, "price":1, "category":1,
                       "csv":1, "preprocessedText":1, "indexedTokens":1,
                       "searchScore": {"$meta":"searchScore"}}},
        {"$limit": 200}
    ]
    return list(mongo_manager.products.aggregate(pipeline))

# =============================================================================
# BM25 (사전 토큰화 활용, 에러 방지)
# =============================================================================
def bm25_search(products, query):
    start = time.time()
    # 사전 토큰화된 필드 사용
    corpus = [p.get("indexedTokens") or [] for p in products]
    if not any(corpus):
        scores = np.zeros(len(products), dtype=float)
    else:
        try:
            bm25 = BM25Okapi(corpus)
            query_tokens = corpus[0].__class__()  # 빈 리스트 대신 함수로 토큰화 필요
            # 실제로는 형태소 분석 함수로 query_tokens 생성
            query_tokens = BM25Okapi.tokenize(query)
            scores = np.array(bm25.get_scores(query_tokens), dtype=float)
            if scores.max() > 0:
                scores /= scores.max()
        except ZeroDivisionError:
            scores = np.zeros(len(products), dtype=float)
    print(f"[BM25 계산 소요 시간]: {time.time()-start:.2f}s")
    return scores

# =============================================================================
# 벡터 유사도 (GPU + 배치)
# =============================================================================
def vector_similarity_search(products, expanded_queries):
    start = time.time()
    img_emb = torch.from_numpy(np.stack([p['imageEmbedding'] for p in products])).cuda()
    txt_emb = torch.from_numpy(np.stack([p['textEmbedding'] for p in products])).cuda()

    e5_list, clip_list = [], []
    for q in expanded_queries:
        e5_list.append(torch.from_numpy(get_text_embedding(q)))
        clip_list.append(torch.from_numpy(get_clip_text_embedding(q)))
    e5_batch = torch.stack(e5_list).cuda()
    clip_batch = torch.stack(clip_list).cuda()

    sim_text = torch.nn.functional.cosine_similarity(e5_batch.unsqueeze(1), txt_emb.unsqueeze(0), dim=2)
    sim_img  = torch.nn.functional.cosine_similarity(clip_batch.unsqueeze(1), img_emb.unsqueeze(0), dim=2)
    scores = (0.6 * sim_text + 0.4 * sim_img).max(dim=0).values.cpu().numpy()
    print(f"[벡터 유사도 계산 소요 시간]: {time.time()-start:.2f}s")
    return scores

# =============================================================================
# LLM 보너스 (preprocessedText 필드 활용)
# =============================================================================
def compute_llm_bonus(products, attrs):
    start = time.time()
    keywords = [v.lower() for v in attrs.values() if v]
    texts = [p.get('preprocessedText','').lower() for p in products]
    bonus = np.array([sum(text.count(k) for k in keywords) for text in texts], dtype=float)
    if bonus.max() > 0:
        bonus /= bonus.max()
    print(f"[LLM 보너스 계산 소요 시간]: {time.time()-start:.2f}s")
    return bonus

# =============================================================================
# 하이브리드 서치 + 병렬 처리
# =============================================================================
def hybrid_search(query, top_k=None):
    start_total = time.time()
    refined, attrs = refined_query_processing(query)
    cat = infer_category(refined, mongo_manager.db)
    if cat and not attrs.get('category'):
        attrs['category'] = cat

    products = atlas_search(refined, attrs)
    if not products:
        return []

    expanded = [refined]  # 예시 단일 확장

    # BM25와 벡터 유사도 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(bm25_search, products, refined)
        f2 = ex.submit(vector_similarity_search, products, expanded)
        bm25_scores = f1.result()
        vec_scores  = f2.result()

    llm_bonus = compute_llm_bonus(products, attrs)

    # 최종 결합
    weights = {'vector':0.5,'bm25':0.3,'llm':0.2}
    final = weights['vector']*vec_scores + weights['bm25']*bm25_scores + weights['llm']*llm_bonus
    print(f"[결합 점수 계산 소요 시간]: {time.time()-start_total - (time.time()-start_total):.2f}s")  # 실제는 별도 측정

    # 임계치 필터
    thr = 0.3
    idxs = np.where(final >= thr)[0]
    idxs = idxs[np.argsort(final[idxs])[::-1]]

    # 결과 조립
    results = []
    for i in idxs[:(top_k or len(idxs))]:
        p = products[i]
        results.append({'이름':p['name'],'설명':p['description'],'링크':p['link'],'유사도':float(final[i])})

    print(f"[검색 총 소요 시간]: {time.time()-start_total:.2f}s")
    return results
