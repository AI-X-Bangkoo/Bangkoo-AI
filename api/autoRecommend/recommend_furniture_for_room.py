import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.image_analysis_utils import analyze_room_with_gemini_by_file
from fastapi import UploadFile, HTTPException

async def recommend_furniture_for_room(
    file: UploadFile,
    style_keywords: list[str] = None,
    min_price: int = None,
    max_price: int = None
):
    """
    업로드된 방 이미지 파일과 스타일 정보를 기반으로 가구를 추천하는 함수
    :param room_image: 방 이미지 파일 (UploadFile)
    :param style_keywords: 추가적인 스타일 키워드 리스트
    :param min_price: 최소 가격 필터
    :param max_price: 최대 가격 필터
    :return: 추천된 가구 리스트
    """
    try:
        # 1) 방 이미지 분석 시작
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

        # 2) 스타일 설명 텍스트 생성
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

        # 3) 텍스트 임베딩 생성
        print("🧠 텍스트 임베딩 생성 시작")
        room_style_embedding = model_manager.text_model.encode([style_desc], normalize_embeddings=True)
        print("🧠 임베딩 생성 완료, shape:", np.shape(room_style_embedding))

        # 4) MongoDB에서 가구 데이터 로드
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
            products.append(doc)
            vectors.append(doc["textEmbedding"])
        print(f"💾 가구 로드 완료, 총 {len(products)}개")

        if not products:
            return [{"이름": "추천 실패", "추천이유": "조건에 맞는 제품이 없습니다."}]

        # 5) 코사인 유사도 계산
        print("🔢 유사도 계산 시작")
        text_vectors = np.array(vectors)
        text_sims = cosine_similarity(room_style_embedding, text_vectors)[0]
        print("🔢 유사도 예시(첫 5개):", text_sims[:5])
        top_indices = text_sims.argsort()[::-1][:10]
        print("🔢 상위 인덱스:", top_indices)

        # 6) 추천 결과 정리
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
        print("❌ 가구 추천 파이프라인 중 오류 발생:", str(e))
        raise HTTPException(status_code=500, detail=f"추천 오류: {str(e)}")