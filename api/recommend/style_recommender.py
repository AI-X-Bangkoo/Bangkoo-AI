import sys
import os
import json
import time
import asyncio
from dotenv import load_dotenv
import random
from pymongo import MongoClient
import google.generativeai as genai
from model_loader import model_manager
from utils.markdown_utils import extract_json_from_markdown

# 경로 설정 및 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mongo_manager import mongo_manager

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.0-flash")

async def style_recommender(styles, min_price=None, max_price=None, top_k=10):
    if not mongo_manager.ready:
        mongo_manager.connect()
    db = mongo_manager.db
    collection = db["products"]

    # --- 스타일 설명 생성 ---
    style_text = ", ".join(styles)
    style_desc = f"이 스타일은 {style_text} 스타일로, 깔끔하고 조화롭고 공간에 어울리는 디자인을 중시합니다."

    # --- 스타일 임베딩 생성 (e5 모델 기반) ---
    query_vec = model_manager.text_model.encode([f"query: {style_desc}"], normalize_embeddings=True)[0]

    # --- MongoDB $vectorSearch 쿼리 구성 ---
    vector_query = {
        "$vectorSearch": {
            "index": "vector_index_v2",  # 사용 중인 벡터 인덱스 이름으로 교체
            "path": "combinedEmbedding",
            "queryVector": query_vec.tolist(),
            "numCandidates": 100,
            "limit": 30
        }
    }

    price_filter = {}
    if min_price:
        price_filter["$gte"] = min_price
    if max_price:
        price_filter["$lte"] = max_price

    match_stage = {"$match": {}}
    if price_filter:
        match_stage["$match"]["price"] = price_filter

    pipeline = [vector_query]
    if match_stage["$match"]:
        pipeline.append(match_stage)
    pipeline.append({"$project": {
        "_id": 0,
        "name": 1,
        "description": 1,
        "detail": 1,
        "price": 1,
        "link": 1,
        "imageUrl": 1,
        "csv": 1
    }})

    products = list(collection.aggregate(pipeline))

    if not products:
        return [{"이름": "추천 실패", "추천이유": "조건에 맞는 제품이 없습니다."}]

    # --- Gemini 프롬프트 구성 ---
    product_summary = "\n".join([
        f"이름: {p['name']}, 가격: {p['price']}, 링크: {p['link']}, 상세설명: {p.get('detail', '')}, 이미지: {p.get('imageUrl')}"
        for p in products
    ])

    filter_text = f"{min_price:,} ~ {max_price:,}원" if min_price and max_price else "가격 제한 없음"

    prompt = f"""
당신은 인테리어 스타일 기반 추천 전문가입니다.

[사용자 선호 스타일]
{style_text}

[가격 필터]
{filter_text}

[추천 후보 제품 요약]
{product_summary}

[추천 조건]
- '{style_text}' 스타일에 어울리는 제품만 골라주세요.
- 가격대({min_price} ~ {max_price})를 벗어나는 제품은 추천하지 마세요.
- 조건에 맞는 제품을 가능한 많이 추천해주세요 (최대 10개).
- JSON 배열 형식으로 반환하세요.
- 추천이유는 단순 키워드가 아닌, 해당 제품이 왜 이 스타일과 어울리는지 구체적으로 설명해 주세요.
- 예를 들어 디자인 형태, 색상, 재질, 분위기, 배치 시 공간에 주는 느낌 등을 기반으로,
  스타일적 연관성과 감성적 효과를 함께 서술해 주세요.

[반환 예시]
[
  {{
    "이름": "제품명",
    "설명": "간단한 설명",
    "링크": "제품 URL",
    "이미지": "이미지 URL",
    "상세설명": "제품의 특징",
    "가격": 123000,
    "추천이유": "화이트 톤과 직선적 디자인이 미니멀리즘에 어울립니다. 공간을 정돈된 분위기로 만들어주는 데 효과적입니다."
  }}
]
"""

    # Gemini 호출
    start = time.time()
    response = await asyncio.to_thread(model.generate_content, prompt)
    print(f"[DEBUG] Gemini 호출 소요시간: {time.time() - start:.2f}초")

    try:
        parsed_json = extract_json_from_markdown(response.text)
        return json.loads(parsed_json)
    except Exception as e:
        print("[ERROR] Gemini 응답 파싱 실패:", e)
        print("[RESPONSE]", response.text)
        return [{"이름": "추천 실패", "추천이유": "Gemini 응답 파싱 실패"}]