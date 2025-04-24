import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.image_analysis_utils import analyze_room_with_gemini_by_file
from fastapi import UploadFile, HTTPException, APIRouter, File
from typing import List
from pydantic import BaseModel

router = APIRouter()

class RecommendedProduct(BaseModel):
    이름: str
    설명: str
    링크: str
    이미지: str
    가격: str
    추천이유: str

@router.post("/analyze_room", response_model=List[RecommendedProduct])
async def analyze_and_recommend(
    file: UploadFile = File(...),
    style_keywords: List[str] = None,
    min_price: int = None,
    max_price: int = None
):
    try:
        print("🔍 방 이미지 분석 시작")
        room_analysis = await analyze_room_with_gemini_by_file(file)
        print("🔍 분석 결과:", room_analysis)

        room_style = room_analysis.get("style", "unknown")
        color_palette = room_analysis.get("color_palette", [])
        furniture_types = room_analysis.get("furniture_types", [])
        materials = room_analysis.get("materials", [])
        lighting_mood = room_analysis.get("lighting_mood", "")
        layout_features = room_analysis.get("layout_features", "")
        decor_items = room_analysis.get("decor_items", [])

        style_desc_parts = [
            f"style: {room_style}",
            f"colors: {', '.join(color_palette)}",
            f"furniture: {', '.join(furniture_types)}",
            f"materials: {', '.join(materials)}",
            f"mood: {lighting_mood}",
            f"layout: {layout_features}",
            f"decor: {', '.join(decor_items)}"
        ]
        if style_keywords:
            style_desc_parts.append(f"keywords: {', '.join(style_keywords)}")
        style_desc = " | ".join(style_desc_parts)
        print("📝 스타일 설명 텍스트:", style_desc)

        print("🧠 텍스트 임베딩 생성 시작")
        room_style_embedding = model_manager.text_model.encode([style_desc], normalize_embeddings=True)
        print("🧠 임베딩 생성 완료, shape:", np.shape(room_style_embedding))

        if not mongo_manager.ready:
            mongo_manager.connect()
        collection = mongo_manager.db["products"]

        cursor = collection.find(
            {"textEmbedding": {"$exists": True}},
            {"name": 1, "price": 1, "category": 1, "textEmbedding": 1, "link": 1, "imageUrl": 1, "description": 1}
        )

        products = []
        vectors = []
        for doc in cursor:
            price_str = doc.get("price", "").replace(",", "").strip()
            try:
                price = int(price_str) if price_str else 0
            except ValueError:
                price = 0
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue

            embedding = doc.get("textEmbedding")
            if not isinstance(embedding,list) or len(embedding) != 768:
                print(f"잘못된 입베딩 스킵: name={doc.get('name')}, 길이={len(embedding) if isinstance(embedding,list) else 'None'}")
                continue
            products.append(doc)
            vectors.append(embedding)
        print(f"💾 가구 로드 완료, 총 {len(products)}개")

        if not products:
            return [{"이름": "추천 실패", "추천이유": "조건에 맞는 제품이 없습니다."}]

        print("🔢 유사도 계산 시작")
        #임베딩 배열 만들기
        text_vectors = np.array(vectors, dtype=np.float32)
        room_style_embedding = np.array(room_style_embedding,dtype=np.float32).reshape(1,-1)

        #유사도 계산
        text_sims = cosine_similarity(room_style_embedding, text_vectors)[0]
        print("🔢 유사도 예시(첫 5개):", text_sims[:5])


        top_indices = text_sims.argsort()[::-1][:10]
        print("🔢 상위 인덱스:", top_indices)

        recommended_results = []
        for i in top_indices:
            product = products[i]
            recommended_results.append({
                "이름": product["name"],
                "설명": product.get("description", ""),
                "링크": product.get("link", ""),
                "이미지": product.get("imageUrl", ""),
                "가격": product.get("price", ""),
                "추천이유": f"{room_style} 스타일의 특징과 잘 어울리는 {product['category']}입니다."
            })
        print(f"✅ 추천 결과 {len(recommended_results)}개 생성 완료")

        return recommended_results

    except Exception as e:
        print("❌ 분석 및 추천 파이프라인 중 오류 발생:", str(e))
        raise HTTPException(status_code=500, detail=f"분석 및 추천 오류: {str(e)}")
