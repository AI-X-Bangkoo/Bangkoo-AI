import os
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from pymongo import MongoClient
from dotenv import load_dotenv
from model_loader import model_manager
from fastapi import HTTPException
from utils.get_image_caption import get_image_caption_and_embedding
from sklearn.metrics.pairwise import cosine_similarity
from mongo_manager import mongo_manager
import torch

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

    # 임베딩 생성
    image_embedding = get_image_embedding(image)  # (1, 1024)
    caption, text_embedding = get_image_caption_and_embedding(image)  # (1, 768)

    print("[5-3] 이미지 + 캡션 임베딩 생성 완료")

    # 정규화 + 가중치 결합 후 다시 정규화
    image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)
    text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)

    # 가중치 부여
    image_weight = 0.7
    text_weight = 0.3

    combined_query = np.concatenate([
        image_embedding * image_weight,
        text_embedding * text_weight
    ], axis=1)

    # 결합 후 전체 정규화
    combined_query = combined_query / np.linalg.norm(combined_query, axis=1, keepdims=True)
    print("combined_query shape:", combined_query.shape)
    print("combined_query norm:", np.linalg.norm(combined_query))
    print("combined_query example:", combined_query[0][:5])  # 앞 5개만



    # MongoDB 연결
    if not mongo_manager.ready:
        mongo_manager.connect()
    collection = mongo_manager.products

    print("[5-4] 벡터 검색 시작 (MongoDB vectorSearch)")
    try:
        results_cursor = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "combinedEmbedding",
                    "queryVector": combined_query[0].tolist(),
                    "numCandidates": 100,
                    "limit": top_k,
                    "similarity": "cosine"
                }
            }
        ])
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {str(e)}")


    results = []
    for doc in results_cursor:
        print("유사도", doc.get("score", 0.0))
        results.append({
            "이름": doc.get("name"),
            "설명": doc.get("description"),
            "상세설명": doc.get("detail", ""),
            "링크": doc.get("link"),
            "이미지": doc.get("imageUrl"),
            "할인가": doc.get("price", "정보 없음"),
            "정상가": doc.get("price", "정보 없음"),
            "csv": doc.get("csv", ""),
            "유사도": doc.get("score", 0.0)  # MongoDB vectorSearch는 자동으로 score 필드를 붙여줌
        })

    print(f"[5-6] 결과 생성 완료 (Top {top_k})")
    return results
     