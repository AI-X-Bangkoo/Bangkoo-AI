from model_loader import model_manager
from mongo_manager import mongo_manager
from datetime import datetime
from pymongo import UpdateOne
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Gemini API 호출 함수 예시
def generate_details_from_gemini(name, description, price, image_url, category):
    prompt = f"""
다음은 IKEA 가구에 대한 정보입니다:

- 이름: {name}
- 설명: {description}
- 가격: {price}
- 이미지: {image_url}
- 카테고리: {category}

이 정보를 바탕으로 다음 항목을 생성해주세요:

1. 이 가구에 대한 시각적 외형과 스타일 분석 포함한 상세 설명 (마크다운 형식 OK)
2. 이 제품에 맞는 키워드 기반 인테리어 카테고리 1개 (예: 가전제품, 소파, 의자, 수납장, 테이블 등)
"""
    # 이 부분은 실제 Gemini API에 맞게 구현 필요 (예시로 작성)
    headers = {
        "Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}",
        "Content-Type": "application/json"
    }
    response = requests.post("https://your-gemini-api-endpoint.com/generate", headers=headers, json={"prompt": prompt})
    
    if response.status_code == 200:
        result = response.json()
        return result.get("상세설명"), result.get("키워드카테고리")
    else:
        print("Gemini API 오류:", response.text)
        return "", ""

def prepare_product_document(user_input: dict):
    name = user_input["이름"]
    description = user_input["설명"]
    price = user_input["할인가"]
    link = user_input["링크"]
    image_url = user_input["이미지"]
    category = user_input["키워드카테고리"]

    상세설명, 키워드카테고리 = generate_details_from_gemini(name, description, price, image_url, category)

    # MongoDB용 문서 생성
    item = {
        "name": name,
        "description": description,
        "price": price,
        "originalPrice": user_input["정상가"],
        "link": link,
        "imageUrl": image_url,
        "csv": user_input.get("csv", "ikea_products_generated.csv"),
        "detail": 상세설명,
        "category": category,
        "키워드카테고리": 키워드카테고리,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }

    # 임베딩 추가
    item = encode_embeddings(item)

    return item

def upload_single_product(item: dict):
    collection = mongo_manager.products
    mongo_manager.connect()
    collection.update_one(
        {"link": item["link"]},
        {"$set": item, "$setOnInsert": {"createdAt": item["createdAt"]}},
        upsert=True
    )
    print(f"{item['name']} 업로드 완료")

# 테스트
if __name__ == "__main__":
    sample_input = {
        "이름": "LAGAN 라간",
        "설명": "빌트인 식기세척기, 60 cm",
        "링크": "https://www.ikea.com/kr/ko/p/lagan-integrated-dishwasher-40568019/",
        "이미지": "https://www.ikea.com/kr/ko/images/products/lagan-integrated-dishwasher__1209974_pe909504_s5.jpg?f=u",
        "할인가": "699,000",
        "정상가": "699,000",
        "키워드카테고리": "가전제품"
    }

    doc = prepare_product_document(sample_input)
    upload_single_product(doc)
