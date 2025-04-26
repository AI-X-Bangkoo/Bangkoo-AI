import os
import io
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from fastapi import HTTPException
from model_loader import model_manager
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mongo_manager import mongo_manager
from utils.get_image_caption import get_image_caption_and_embedding, get_image_embedding
from utils.visual_color_utils import get_color_keywords_from_db
from utils.query_utils import extract_color_from_caption, extract_shape_from_caption

load_dotenv()


# =================================================================
# MongoDB Atlas Search의 $vectorSearch를 사용해 combinedEmbedding 필드에서 쿼리 벡터와 유사한 문서를 조회
# 매개변수:
#       - query_vector: 1792차원 결합 임베딩 (list of float)
#       - top_k: 반환할 상위 개수
#       - category: (선택) 카테고리 필터. 주어지면 벡터 검색 시 필터로 적용.

#     반환값:
#       - 검색 결과 문서 리스트. 각 문서는 score 메타정보를 포함.
# =================================================================
def perform_vector_search_combined(query_vector: list,
                                   top_k: int = 50,
                                   category: Optional[str] = None):
    collection = mongo_manager.products

    # 1) 벡터 검색 스테이지 (첫 번째)
    vec_stage = {
        "$vectorSearch": {
            "index": "vector_index_v2",
            "path": "combinedEmbedding",
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": top_k,
            "similarity": "cosine",
            # category 필터가 있을 때만 단순 match 문서로 추가
            **({"filter": {"category": category}} if category else {})
        }
    }

    # 2) projection
    proj_stage = {
        "$project": {
            "_id": 0,
            "name": 1,
            "description": 1,
            "detail": 1,
            "link": 1,
            "imageUrl": 1,
            "price": 1,
            "category": 1,
            "csv": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
    }

    try:
        return list(collection.aggregate([vec_stage, proj_stage]))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {e}")

# =================================================================
# 사용자가 업로드한 이미지에 대해 다음 단계를 수행:

    # 1. 이미지 디코딩 및 모델 준비 확인
    # 2. 이미지 임베딩과 캡션/텍스트 임베딩, 카테고리 추출
    # 3. combinedEmbedding 벡터 생성
    # 4. MongoDB에서 $vectorSearch를 사용해 유사 제품 검색
    # 5. 캡션 기반 shape/color 키워드로 소프트 필터링
    # 6. JSON-friendly 포맷으로 결과 반환

    # 매개변수:
    #   - contents: 업로드된 이미지 바이트
    #   - top_k: 반환할 결과 개수

    # 반환값:
    #   - 제품 검색 결과 리스트 (이름, 설명, 링크, 이미지 URL, 가격, 유사도 등)
# =================================================================
def image_search(contents: bytes, top_k: int = 50):
    if not model_manager.ready:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다. JPEG/PNG 권장")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 디코딩 실패: {e}")

    # 이미지 임베딩, 캡션+텍스트 임베딩, 카테고리 추출
    image_emb = get_image_embedding(image)[0].astype(np.float32)       # (1024,)
    caption, text_emb_arr, category = get_image_caption_and_embedding(image)
    text_emb  = text_emb_arr[0].astype(np.float32)                     # (768,)
    combined_query = np.concatenate([image_emb, text_emb], axis=-1).tolist()  # (1792,)

    # shape/color 키워드
    shape_key, shape_syns = extract_shape_from_caption(caption, mongo_manager.db)
    color_key = extract_color_from_caption(caption)
    color_dict = get_color_keywords_from_db()
    color_syns = color_dict.get(color_key, []) if color_key else []

    # DB 연결 혹시 몰라서
    if not mongo_manager.ready:
        mongo_manager.connect()

    cat = category.strip().lower()
    # print(f"[검색 시작] 캡션={caption!r}, 카테고리={cat!r}, shape={shape_key}, color={color_key}")

    # combinedEmbedding으로 vectorSearch
    results = perform_vector_search_combined(
        query_vector=combined_query,
        top_k=top_k,
        category=cat or None
    )

    # shape/color 후처리 필터링
    filtered = []
    for doc in results:
        text_blob = f"{doc['name']} {doc['description']} {doc.get('detail','')}".lower()
        # shape 필터
        if shape_syns and not any(w in text_blob for w in shape_syns):
            continue
        # color 필터
        if color_syns and not any(c in text_blob or c in caption.lower() for c in color_syns):
            continue
        filtered.append(doc)

    final = filtered if filtered else results

    # 결과 포맷
    output = []
    for d in final:
        output.append({
            "이름":     d["name"],
            "설명":     d["description"],
            "상세설명": d.get("detail",""),
            "링크":     d["link"],
            "이미지":   d.get("imageUrl"),
            "할인가":   d.get("price","정보 없음"),
            "정상가":   d.get("price","정보 없음"),
            "csv":      d.get("csv",""),
            "유사도":   float(d["score"])
        })

    print(f"[검색 완료] 총 {len(output)}건")
    return output
