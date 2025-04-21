import os
import io
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from fastapi import HTTPException
from model_loader import model_manager
from typing import Optional
from rank_bm25 import BM25Okapi

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mongo_manager import mongo_manager
from utils.get_image_caption import get_image_caption_and_embedding, get_image_embedding
from utils.visual_color_utils import (
    # rerank_by_visual_similarity,
    extract_color_token,
    apply_color_bonus,
    get_color_keywords_from_db
)
from utils.query_utils import extract_color_from_caption, extract_shape_from_caption, get_shape_keywords_from_db


"""
최초 작성자: 김동규
최초 작성일: 2025-04-02

업로드된 이미지에서 이미지 임베딩과 캡션/텍스트 임베딩을 추출하고,
학습 가능한 Fusion 네트워크를 통해 결합한 후,
MongoDB Atlas Search의 $vectorSearch를 통해 유사 제품 검색을 수행하는 모듈.
도미넌트 색상 기반 필터링은 제거됨.
"""

load_dotenv()

# =============================================================================
# 벡터 서치 aggregate
# =============================================================================
def perform_vector_search_combined(
    query_vector: list,
    top_k: int = 50,
    category: Optional[str] = None
):
    collection = mongo_manager.products

    vec_stage = {
        "$vectorSearch": {
            "index": "vector_index_v2",
            "path":  "combinedEmbedding",      # <-- combinedEmbedding만 사용
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": top_k,
            "similarity": "cosine",
            **({"filter": {"category": category}} if category else {})
        }
    }
    pipeline = [
        vec_stage,
        {"$project": {
            "_id": 0,
            "name":1, "description":1, "detail":1,
            "link":1, "imageUrl":1, "price":1,
            "category":1, "csv":1,
            "score": {"$meta": "vectorSearchScore"}
        }}
    ]

    try:
        results = list(collection.aggregate(pipeline))
        print(f"검색 결과 수: {len(results)} (combinedEmbedding, category={category})")
        return results
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {e}")

# =============================================================================
# 이미지 디코딩 및 이미지 기반 유사도 검색
# =============================================================================
def image_search(contents: bytes, top_k: int = 50):
    # 1. 이미지 로드
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다. JPEG/PNG 권장")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 디코딩 실패: {e}")

    # 2. 임베딩 및 캡션 생성
    image_emb = get_image_embedding(image)[0].astype(np.float32)          # (1024,)
    caption, text_emb_array, category = get_image_caption_and_embedding(image)
    text_emb  = text_emb_array[0].astype(np.float32)                     # (768,)

    # 3. 쿼리용 combined vector 생성
    query_combined = np.concatenate([image_emb, text_emb], axis=-1).tolist()  # (1792,)

    # 4. MongoDB 연결 보장
    if not mongo_manager.ready:
        mongo_manager.connect()

    # 5. combinedEmbedding 만으로 한 번 검색
    cat = category.strip().lower() if isinstance(category, str) else None
    results = perform_vector_search_combined(
        query_vector=query_combined,
        top_k=top_k,
        category=cat
    )

    # 6. soft 필터링 & 결과 구성
    final = []
    for r in results:
        # (기존 Category / Color / Shape 필터 로직)
        prod_cat = r['category'].strip().lower()
        if cat and cat not in prod_cat and prod_cat not in cat:
            continue

        color_key = extract_color_from_caption(caption)
        if color_key:
            syns = get_color_keywords_from_db().get(color_key, [])
            txt = f"{r['name']} {r['description']} {r.get('detail','')}".lower()
            if syns and not any(s in txt or s in caption.lower() for s in syns):
                continue

        shape_key, shape_syns = extract_shape_from_caption(caption, mongo_manager.db)
        if shape_key and shape_syns:
            txt = f"{r['name']} {r['description']} {r.get('detail','')}".lower()
            # soft apply: 하나라도 match 되면 통과
            if not any(w in txt for w in shape_syns):
                # 매칭 문서가 아예 없으면 필터 건너뛰고 추가
                pass

        final.append({
            '이름':    r['name'],
            '설명':    r['description'],
            '상세설명': r.get('detail',''),
            '링크':    r['link'],
            '이미지':  r.get('imageUrl'),
            '할인가':  r.get('price','정보 없음'),
            '정상가':  r.get('price','정보 없음'),
            'csv':     r.get('csv',''),
            '유사도':  float(r['score'])
        })

    return final
