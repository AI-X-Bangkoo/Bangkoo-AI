import os
import io
import numpy as np
from PIL import Image
import torch
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from model_loader import model_manager
from PIL import UnidentifiedImageError
from fastapi import HTTPException
import pillow_avif
import time
from api.search.hybrid_search import infer_category

# 캐시 리셋 DB에 제품 추가할 수 있으니
# last_cache_time = None
# from datetime import datetime, timedelta
# if last_cache_time is None or datetime.now() - last_cache_time > timedelta(minutes=5):
#     cached_products = None
#     cached_image_embeddings = None
#     last_cache_time = datetime.now()


load_dotenv()

cached_products = None
cached_image_embeddings = None

"""
최초 작성자: 김동규
최초 작성일: 2025-04-07

이미지 기반 검색 모듈
- 모델이 로드되지 않았을 경우 예외 처리
- 모델은 함수 내에서 동적으로 접근
"""

def get_image_embedding(image: Image.Image):
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")

    clip_model = model_manager.clip_model
    clip_processor = model_manager.clip_processor
    device = model_manager.device

    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def fast_cosine_similarity(query, embeddings):
    print("[fast_sim] 유사도 계산 시작")
    query = query.astype(np.float32)
    embeddings = embeddings.astype(np.float32)
    print("[fast_sim] 타입 변환 완료")

    dot = np.dot(embeddings, query.T).squeeze()
    print("[fast_sim] dot 연산 완료")

    query_norm = np.linalg.norm(query)
    emb_norm = np.linalg.norm(embeddings, axis=1)
    print("[fast_sim] norm 계산 완료")

    result = dot / (emb_norm * query_norm + 1e-8)
    print("[fast_sim] 최종 유사도 계산 완료")
    return result


def image_search(contents: bytes, top_k=10):
    global cached_products, cached_image_embeddings

    print("[5-1] 이미지 디코딩 시작")
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("[5-2] 이미지 디코딩 성공")
    except UnidentifiedImageError:
        print("이미지 포맷 인식 실패 (지원하지 않는 형식)")
        raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다. JPEG, PNG 권장")
    except Exception as e:
        print(f"이미지 디코딩 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 디코딩 실패: {str(e)}")

    query_embedding = get_image_embedding(image)
    print("[5-3] 이미지 임베딩 생성 완료")
    print(f"[5-3-1] query shape: {query_embedding.shape}")

    # DB + 임베딩 캐시 적용
    if cached_products is None or cached_image_embeddings is None:
        print("[5-3-1.5] DB에서 product 목록 조회 시작")
        start = time.time()
        MONGO_URI = os.getenv("MONGO_URI")
        
        # 한 번만 호출 하도록
        client = MongoClient(MONGO_URI)
        
        db = client["bangkoo"]
        cached_products = []
        print("[5-3-1.5.1] 제품 하나씩 순회 시작")
        cursor = db["products"].find({}, {
            "name": 1,
            "description": 1,
            "detail": 1,
            "link": 1,
            "imageUrl": 1,
            "price": 1,
            "csv": 1,
            "imageEmbedding": 1
        })

        cached_products = list(cursor)
        # for idx, p in enumerate(cursor):
        #     if idx < 5:
        #         print(f"  - 제품 {idx+1}: {p.get('name', '이름 없음')} (ID: {p.get('_id')})")
        #     elif idx == 5:
        #         print("  ... 생략 중 ...")
        #     cached_products.append(p)

        # print(f"[5-3-1.5.2] 총 {len(cached_products)}개 제품 수집 완료")

        end = time.time()
        print(f"DB 조회 완료: {len(cached_products)}개, 소요 시간: {end - start:.2f}초")
        print(f"[5-3-1.6] DB에서 product {len(cached_products)}개 조회됨")
        
        for i, p in enumerate(cached_products[:5]):
            print(f"  - 제품 {i+1}: {p.get('name', '이름 없음')} (ID: {p.get('_id')})")

        try:
            cached_image_embeddings = np.array([p["imageEmbedding"] for p in cached_products], dtype=np.float32)
        except Exception as e:
            print(f"[오류] image_embeddings 변환 중 예외 발생: {e}")
            raise HTTPException(status_code=500, detail=f"DB 임베딩 변환 실패: {str(e)}")

        print(f"[5-3-2] db embeddings shape: {cached_image_embeddings.shape}")
    else:
        print("[5-3-2] 캐시된 임베딩 사용 중")

    print("[5-4] 유사도 계산 시작")
    sim = fast_cosine_similarity(query_embedding, cached_image_embeddings)
    print("[5-5] 유사도 계산 완료")

    top_idx = np.argsort(sim)[::-1][:top_k]

    results = []
    for i in top_idx:
        item = cached_products[i]
        results.append({
            "이름": item["name"],
            "설명": item["description"],
            "상세설명": item.get("detail", ""),
            "링크": item["link"],
            "이미지": item["imageUrl"],
            "할인가": item.get("price", "정보 없음"),
            "정상가": item.get("price", "정보 없음"),
            "csv": item.get("csv", ""),
            "유사도": float(sim[i])
        })

    print(f"[5-6] 결과 생성 완료 (Top {top_k})")
    return results
