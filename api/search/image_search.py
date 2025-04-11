import os
import io
import sys
import time
import base64
import numpy as np
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from fastapi import HTTPException
from model_loader import model_manager
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mongo_manager import mongo_manager
from utils.get_image_caption import get_image_caption_and_embedding, get_image_embedding, extract_color_from_caption
from utils.fusion_network import FusionNetwork 
from utils.visual_color_utils import (
    # rerank_by_visual_similarity,
    extract_color_token,
    apply_color_bonus
)


"""
최초 작성자: 김동규
최초 작성일: 2025-04-02

업로드된 이미지에서 이미지 임베딩과 캡션/텍스트 임베딩을 추출하고,
학습 가능한 Fusion 네트워크를 통해 결합한 후,
MongoDB Atlas Search의 $vectorSearch를 통해 유사 제품 검색을 수행하는 모듈.
도미넌트 색상 기반 필터링은 제거됨.
"""

load_dotenv()

# fusion 네트워크 인스턴스 생성
fusion_net = FusionNetwork(1024, 768, 1792)

def perform_vector_search(query_vector: list, top_k=50):
    collection = mongo_manager.products
    try:
        results_cursor = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index_v2",
                    "path": "combinedEmbedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": top_k,
                    "similarity": "cosine"
                }
            },
            {
                "$project": {
                    "name": 1,
                    "description": 1,
                    "detail": 1,
                    "link": 1,
                    "imageUrl": 1,
                    "price": 1,
                    "category": 1,
                    "csv": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "combinedEmbedding": 1
                }
            }
        ])
        results = list(results_cursor)
        print(f"검색 결과 수: {len(results)}")
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {str(e)}")

def image_search(contents: bytes, top_k=10):
    print("[5-1] 이미지 디코딩 시작")
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("[5-2] 이미지 디코딩 성공")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다. JPEG, PNG 권장")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 디코딩 실패: {str(e)}")

    # --- 임베딩 및 캡션 생성 ---
    image_embedding = get_image_embedding(image)  # (1, 1024)
    caption, text_embedding, category = get_image_caption_and_embedding(image)  # (1, 768), category
    print("[5-3] 임베딩 및 캡션/텍스트 임베딩 생성 완료")
    print(f"[5-3] 생성된 캡션: {caption}")
    print(f"[5-3] 추출된 카테고리: {category}")

    # --- Fusion 네트워크 적용 ---
    img_tensor = torch.from_numpy(image_embedding).float()  # (1, 1024)
    txt_tensor = torch.from_numpy(text_embedding).float()   # (1, 768)
    fused_tensor = fusion_net(img_tensor, txt_tensor)  # (1, 1792)
    combined_query = fused_tensor.detach().numpy()

    print("combined_query shape:", combined_query.shape)
    print("combined_query norm:", np.linalg.norm(combined_query))
    print("combined_query example:", combined_query[0][:5])

    # --- MongoDB 벡터 검색 ---
    if not mongo_manager.ready:
        mongo_manager.connect()
    collection = mongo_manager.products
    total_docs = collection.count_documents({})
    docs_with_embedding = collection.count_documents({"combinedEmbedding": {"$exists": True}})
    print("총 문서 수:", total_docs)
    print("combinedEmbedding 있는 문서 수:", docs_with_embedding)
    print("[5-4] 벡터 검색 시작 (MongoDB vectorSearch)")
    try:
        combined_query = combined_query.astype(np.float32)
        query_vector = combined_query[0].tolist()
        results_sorted = perform_vector_search(query_vector, top_k=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {str(e)}")

    # --- 추가 카테고리 필터링 ---
    filtered_results = []
    target_category = category.strip().lower() if isinstance(category, str) else ""
    for doc in results_sorted:
        product_category = doc.get("category", "").strip().lower()
        if target_category and target_category in product_category:
            filtered_results.append(doc)
    if not filtered_results:
        print("검색 결과 없음 (카테고리 필터 포함)")
    else:
        results_sorted = filtered_results

    # --- 색상 보너스 후처리 ---
    try:
        color_key = extract_color_from_caption(caption)
        print(f"[COLOR] 추출된 색상 키: {color_key}")
        results_sorted = apply_color_bonus(results_sorted, color_key)
    except Exception as e:
        print(f"[COLOR] 색상 후처리 실패: {e}")

    # --- 시각적 유사도 기반 재정렬 ---
    # try:
    #     results_sorted = rerank_by_visual_similarity(results_sorted, image_embedding)
    # except Exception as e:
    #     print(f"[RERANK] 시각 유사도 재정렬 실패: {e}")

    # --- 최종 정렬 및 결과 구성 ---
    results_sorted.sort(key=lambda x: x.get("score", 0), reverse=True)
    final_results = []
    for doc in results_sorted[:top_k]:
        print(f"유사도 (최종 score): {doc.get('name')} → {doc.get('score'):.4f}")
        final_results.append({
            "이름": doc.get("name"),
            "설명": doc.get("description"),
            "상세설명": doc.get("detail", ""),
            "링크": doc.get("link"),
            "이미지": doc.get("imageUrl"),
            "할인가": doc.get("price", "정보 없음"),
            "정상가": doc.get("price", "정보 없음"),
            "csv": doc.get("csv", ""),
            "유사도": float(doc.get("score", 0))
        })

    if not final_results:
        print("검색 결과 없음 (최종 결과 부족)")
    print(f"[5-6] 결과 생성 완료 (Top {top_k})")
    return final_results