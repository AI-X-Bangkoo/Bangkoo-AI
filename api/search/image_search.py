import os
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from fastapi import HTTPException

import torch
from sklearn.metrics.pairwise import cosine_similarity

from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.get_image_caption import get_image_caption_and_embedding

load_dotenv()


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

    # 1. 임베딩 추출
    image_embedding = get_image_embedding(image)
    caption, text_embedding, category = get_image_caption_and_embedding(image)
    print("[5-3] 이미지 + 캡션 임베딩 생성 완료")

    # 2. 임베딩 정규화 및 결합
    image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)
    text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)

    image_weight = 0.3
    text_weight = 0.7

    combined_query = np.concatenate([
        image_embedding * image_weight,
        text_embedding * text_weight
    ], axis=1)
    combined_query = combined_query / np.linalg.norm(combined_query, axis=1, keepdims=True)

    print("combined_query shape:", combined_query.shape)
    print("combined_query norm:", np.linalg.norm(combined_query))
    print("combined_query example:", combined_query[0][:5])

    # 3. MongoDB 접속
    if not mongo_manager.ready:
        mongo_manager.connect()
    collection = mongo_manager.products

    print("총 문서 수:", collection.count_documents({}))
    print("combinedEmbedding 있는 문서 수:", collection.count_documents({"combinedEmbedding": {"$exists": True}}))

    # 4. MongoDB 벡터 검색
    print("[5-4] 벡터 검색 시작 (MongoDB vectorSearch)")
    try:
        combined_query = combined_query.astype(np.float32)
        query_vector = combined_query[0].tolist()

        results_cursor = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index_v2",
                    "path": "combinedEmbedding",
                    "queryVector": query_vector,
                    "numCandidates": 1000,
                    "limit": 50,
                    "similarity": "cosine"
                }
            }
        ])

        results_raw = list(results_cursor)
        print(f"검색 결과 수: {len(results_raw)}")

        results_sorted = sorted(
            results_raw,
            key=lambda d: -float(np.dot(query_vector, np.array(d["combinedEmbedding"], dtype=np.float32)) /
                                 (np.linalg.norm(query_vector) * np.linalg.norm(np.array(d["combinedEmbedding"], dtype=np.float32))))
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {str(e)}")

    # 5. 카테고리 필터링
    filtered_results = []
    for doc in results_sorted:
        product_category = doc.get("category", "")
        if product_category == category:
            filtered_results.append(doc)

    # 6. 결과 구성
    results = []
    for doc in filtered_results[:top_k]:
        db_vec = np.array(doc["combinedEmbedding"], dtype=np.float32)
        score = float(np.dot(query_vector, db_vec) /
                      (np.linalg.norm(query_vector) * np.linalg.norm(db_vec)))
        print(f"유사도 재계산: {doc.get('name')} → {score:.4f}")
        results.append({
            "이름": doc.get("name"),
            "설명": doc.get("description"),
            "상세설명": doc.get("detail", ""),
            "링크": doc.get("link"),
            "이미지": doc.get("imageUrl"),
            "할인가": doc.get("price", "정보 없음"),
            "정상가": doc.get("price", "정보 없음"),
            "csv": doc.get("csv", ""),
            "유사도": score
        })

    if not results:
        print("검색 결과 없음 (카테고리 필터 포함 Top K 결과 부족할 수 있음)")

    print(f"[5-6] 결과 생성 완료 (Top {top_k})")
    return results
