import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.image_utils import extract_image_embedding

async def recommend_furniture_for_room(room_image, style_keywords, min_price=None, max_price=None):
    # 1. 방 이미지 분석: 스타일 및 가구 카테고리 추출
    room_style, furniture_categories = await analyze_room_image(room_image)
    
    # 2. 스타일 및 카테고리 텍스트 임베딩 생성
    style_desc = " ".join(style_keywords)
    room_style_embedding = model_manager.text_model.encode([f"room style: {room_style}, {style_desc}"], normalize_embeddings=True)
    
    # 3. 이미지 임베딩 생성 (방 이미지 분석 결과)
    room_image_embedding = await extract_image_embedding(room_image)

    # 4. MongoDB에서 가구 데이터 가져오기
    if not mongo_manager.ready:
        mongo_manager.connect()
    db = mongo_manager.db
    collection = db["products"]
    
    cursor = collection.find({"textEmbedding": {"$exists": True}, "imageEmbedding": {"$exists": True}}, 
                             {"name": 1, "price": 1, "category": 1, "textEmbedding": 1, "imageEmbedding": 1, "link": 1, "imageUrl": 1})
    
    products = []
    vectors = []
    image_vectors = []
    
    for doc in cursor:
        price = int(doc["price"].replace(",", "").strip()) if doc.get("price") else 0
        if min_price and price < min_price: continue
        if max_price and price > max_price: continue

        products.append(doc)
        vectors.append(doc["textEmbedding"])
        image_vectors.append(doc["imageEmbedding"])

    if not products:
        return [{"이름": "추천 실패", "추천이유": "조건에 맞는 제품이 없습니다."}]
    
    # 5. 코사인 유사도 계산 (텍스트 임베딩 및 이미지 임베딩)
    text_vectors = np.array(vectors)
    image_vectors = np.array(image_vectors)
    
    text_sims = cosine_similarity(room_style_embedding, text_vectors)[0]
    image_sims = cosine_similarity(room_image_embedding, image_vectors)[0]
    
    # 두 유사도 점수의 평균으로 가구를 순위 매김
    combined_sims = (text_sims + image_sims) / 2
    top_indices = combined_sims.argsort()[::-1][:10]  # 상위 10개 추천
    
    recommended_products = [products[i] for i in top_indices]

    # 6. 추천 이유 생성
    recommended_results = []
    for product in recommended_products:
        recommended_results.append({
            "이름": product["name"],
            "설명": product["description"],
            "링크": product["link"],
            "이미지": product["imageUrl"],
            "가격": product["price"],
            "추천이유": f"{room_style} 스타일에 어울리는 {product['category']} 제품입니다. 감성적으로 공간에 잘 맞는 디자인입니다."
        })

    return recommended_results
