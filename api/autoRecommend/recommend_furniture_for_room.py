import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.gemini_utils import analyze_room_with_gemini_by_file
from fastapi import UploadFile
from io import BytesIO
from PIL import Image


async def recommend_furniture_for_room(room_image: UploadFile, style_keywords: list[str] = None, min_price: int = None, max_price: int = None):
    """
    업로드된 방 이미지 파일과 스타일 정보를 기반으로 가구를 추천하는 함수
    :param room_image: 방 이미지 파일 (UploadFile)
    :param style_keywords: 추가적인 스타일 키워드 리스트
    :param min_price: 최소 가격 필터
    :param max_price: 최대 가격 필터
    :return: 추천된 가구 리스트
    """
    try:
        # 1. Gemini로 이미지 분석하여 스타일 및 기타 정보 추출
        room_analysis = await analyze_room_with_gemini_by_file(room_image)
        room_style = room_analysis.get("style", "unknown")
        color_palette = room_analysis.get("color_palette", [])
        furniture_types = room_analysis.get("furniture_types", [])
        materials = room_analysis.get("materials", [])
        lighting_mood = room_analysis.get("lighting_mood", "")
        layout_features = room_analysis.get("layout_features", "")
        decor_items = room_analysis.get("decor_items", [])

        # 2. 분석된 정보를 바탕으로 스타일 설명 텍스트 구성
        style_desc_parts = [
            f"style: {room_style}",
            f"colors: {', '.join(color_palette)}",
            f"furniture: {', '.join(furniture_types)}",
            f"materials: {', '.join(materials)}",
            f"mood: {lighting_mood}",
            f"layout: {layout_features}",
            f"decor: {', '.join(decor_items)}"
        ]

        # 추가적인 스타일 키워드가 있을 경우 포함
        if style_keywords:
            style_desc_parts.append(f"keywords: {', '.join(style_keywords)}")

        # 스타일 설명 텍스트 결합
        style_desc = " | ".join(style_desc_parts)

        # 3. 텍스트 임베딩 생성 (정규화 포함)
        room_style_embedding = model_manager.text_model.encode([style_desc], normalize_embeddings=True)

        # 4. MongoDB에서 가구 데이터 가져오기
        if not mongo_manager.ready:
            mongo_manager.connect()  # MongoDB 연결 상태 확인 후 연결
        collection = mongo_manager.db["products"]

        # 제품 데이터 가져오기: 텍스트 임베딩이 있는 가구만 조회
        cursor = collection.find(
            {"textEmbedding": {"$exists": True}},  # textEmbedding이 존재하는 문서만 필터링
            {"name": 1, "price": 1, "category": 1, "textEmbedding": 1, "link": 1, "imageUrl": 1, "description": 1}  # 필요한 필드만 조회
        )

        products = []
        vectors = []

        for doc in cursor:
            # 가격 필터링 (가격 문자열을 정수로 변환 후 조건 체크)
            price = doc.get("price", "").replace(",", "").strip()
            try:
                price = int(price) if price else 0
            except ValueError:
                price = 0  # 가격 값이 비정상일 경우 기본값 0으로 처리

            # 가격 필터링 조건 적용
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue

            # 제품 리스트와 임베딩 벡터 리스트에 추가
            products.append(doc)
            vectors.append(doc["textEmbedding"])

        if not products:
            # 조건에 맞는 제품이 없을 경우 메시지 반환
            return [{"이름": "추천 실패", "추천이유": "조건에 맞는 제품이 없습니다."}]

        # 5. 텍스트 임베딩 간 코사인 유사도 계산
        text_vectors = np.array(vectors)
        text_sims = cosine_similarity(room_style_embedding, text_vectors)[0]  # 1차원 벡터 반환
        top_indices = text_sims.argsort()[::-1][:10]  # 상위 10개 제품 추천

        # 6. 추천 결과 정리
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

        return recommended_results

    except Exception as e:
        logging.error(f"가구 추천 오류: {e}")
        raise
