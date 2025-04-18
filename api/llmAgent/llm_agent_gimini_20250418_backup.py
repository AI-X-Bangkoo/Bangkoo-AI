import json
import os
import re
import sys
import time
import asyncio
from tempfile import NamedTemporaryFile, gettempdir
from fastapi import UploadFile
from dotenv import load_dotenv
from pymongo import MongoClient
from collections import Counter
import google.generativeai as genai
from utils.markdown_utils import extract_json_from_markdown

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mongo_manager import mongo_manager
from utils.keyword_module import (
    extract_keywords_from_query,
    guess_category_from_keywords,
    filter_products_by_category,
    filter_by_query_keywords
)

"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

Gemini 기반 AI 추천 모듈
  - Gemini 호출을 비동기로 처리하며, 결과는 캐싱
  - 쿼리 키워드 추출, 카테고리 추론 및 필터링은 별도 모듈(keyword_module.py)에서 처리
  - 재랭킹 단계에서 프롬프트 길이를 줄여 처리 시간을 단축
"""

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

if not mongo_manager.ready:
    mongo_manager.connect()

db = mongo_manager.db
product_collection = mongo_manager.products

# =============================================================================
# 캐시 디렉터리 설정: OS 임시 디렉터리 하위에 room_style_cache 폴더 생성
# =============================================================================
CACHE_DIR = os.path.join(gettempdir(), "room_style_cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cached_room_style(image_path: str) -> str:
    cache_path = os.path.join(CACHE_DIR, f"room_style_{os.path.basename(image_path)}.txt")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def cache_room_style(image_path: str, description: str) -> str:
    cache_path = os.path.join(CACHE_DIR, f"room_style_{os.path.basename(image_path)}.txt")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(description)
    return description

# =============================================================================
# 비동기 Gemini 호출 (방 스타일 설명 생성)
# =============================================================================
async def get_room_style_description_async(image_path: str) -> str:
    start_time = time.time()
    cached = get_cached_room_style(image_path)
    if cached:
        print(f"[DEBUG] 캐시에서 방 스타일 불러옴 (소요시간: {time.time() - start_time:.2f}초)")
        return cached
    prompt = """
        아래 방 사진을 보고, 스타일 전문가처럼 객관적인 설명을 작성해 주세요.

        - 방의 **전체적인 스타일**, **지배적인 색상**, **공간 분위기**, **구성 요소**를  
        2~3문장의 자연어 설명문으로 작성해 주세요.

        - 이 설명은 추후 **Semantic Search (의미 기반 검색)**에 사용될 예정입니다.  
        따라서 문장 속에 실제 이미지에 보이는 **스타일, 분위기, 색상, 배치 정보** 등이 풍부하게 담겨야 합니다.

        - 키워드 요약은 금지하고, 스타일 설명 문장만 작성해 주세요.

        - 예시:
        "이 방은 밝은 원목 마루와 흰 벽면을 중심으로 구성되어 있어, 북유럽 스타일의 심플하고 따뜻한 느낌을 줍니다.  
        직선형의 간결한 가구들이 균형 있게 배치되어 있어 전체적인 공간은 정돈된 분위기를 가집니다."

        - 주관적인 표현, 추측, 모호한 말은 금지하며
        **실제 이미지에 보이는 시각적 정보만** 바탕으로 작성해 주세요.
        """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = await asyncio.to_thread(model.generate_content, [prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
    description = response.text.strip()
    cache_room_style(image_path, description)
    print(f"[DEBUG] Gemini 방 스타일 설명 생성 완료 (소요시간: {time.time() - start_time:.2f}초)")
    return description

# =============================================================================
# 재랭킹 함수: 후보 제품 정보를 간략하게 요약하고 Gemini 호출을 비동기로 처리
# =============================================================================
async def rerank_ai_recommendations_async(
    room_style: str,
    query: str,
    candidate_products: list,
    previous_results=None,
    min_price=None,
    max_price=None,
    keyword=None,
    style=None,
    category=None
) -> str:
    # 각 후보 제품을 간략히 요약: 이름, 가격, 링크, 상세설명, 이미지 포함
    product_summary = "\n".join([
        f"이름: {p['name']}, 가격: {p.get('price', '정보 없음')}, 링크: {p['link']}, 상세설명: {p['detail']}, 이미지: {p['imageUrl']}"
        for p in candidate_products
    ])

    filter_info = ""
    if min_price is not None and max_price is not None:
        filter_info += f"- 가격대: {min_price:,} ~ {max_price:,}원\n"
    if keyword:
        filter_info += f"- 키워드: {keyword}\n"
    if style:
        filter_info += f"- 스타일: {style}\n"
    if category:
        filter_info += f"- 제품 종류: {category}\n"

    prompt = f"""
당신은 인테리어 전문가입니다.

[방 스타일 설명]
{room_style}

[사용자 요청]
{query}
"""
    if filter_info:
        prompt += f"\n[필터 조건]\n{filter_info}"
    if previous_results:
        prev_json = json.dumps(previous_results, ensure_ascii=False, indent=2)
        prompt += f"\n[이전 추천 결과]\n{prev_json}\n위 결과를 참고하여 다시 3개 이상의 제품을 추천해 주세요."
    if category:
        prompt += f"""
- 반드시 '{category}'에 해당하는 제품만 추천해 주세요.
- '{category}'가 아닌 제품은 '해당 카테고리가 아니므로 제외'라고 이유를 적어주세요.
- 링크/이미지/이름 등을 임의로 작성하지 마세요. 목록에 있는 제품만 그대로 추천해 주세요.
"""
    prompt += f"""
[추천 후보 제품 요약]
{product_summary}

[추천 후보 제품 목록]
아래 제품들은 실제 데이터베이스에서 검색된 후보입니다.
이 중에서 방 스타일과 사용자 요청, 필터 조건에 가장 적합한 3개 이상의 제품을 고르고 JSON 배열로 반환하세요. 조건 미충족 시 "추천이유"에 이유를 써 주세요
형식:
[
  {{
    "이름": "...",
    "설명": "...",
    "링크": "...",
    "이미지": "...",
    "상세설명": "...",
    "할인가": "...",
    "정상가": "...",
    "csv": "...",
    "추천이유": "..."
  }},
  ...
]
"""
    start_time = time.time()
    # 비동기로 Gemini 호출: 재랭킹 프롬프트 처리
    response = await asyncio.to_thread(model.generate_content, [prompt])
    elapsed = time.time() - start_time
    print(f"[DEBUG] Gemini 응답 소요시간 (순수 모델): {elapsed:.2f}초")
    print("[DEBUG] Gemini 재랭킹 응답:\n", response.text)
    return response.text.strip()

# =============================================================================
# 필터를 가지고 제품 추천 함수
# =============================================================================
async def recommend_with_ai_agent(
    image_file: UploadFile,
    query: str,
    min_price: int = None,
    max_price: int = None,
    keyword: str = None,
    style: str = None
):
    overall_start = time.time()
    print(f"[DEBUG] /search 진입 - 전체 요청 form 데이터: {{'query': '{query}'}}")
    
    # 이미지 파일을 임시 파일로 저장
    temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(await image_file.read())
    temp_file.close()
    print(f"[DEBUG] 임시 이미지 저장 완료: {temp_file.name}")

    # --- 비동기로 방 스타일 설명 생성 (여기서 캐싱 사용함) ---
    room_style = await get_room_style_description_async(temp_file.name)

    # --- DB 후보 제품 조회: 최대 1000개로 제한 -> 500개 테스트도 해봐야 할 듯? (속토 측면을 위해) ---
    candidate_limit = 1000
    db_query_start = time.time()
    all_products = list(product_collection.find(
        {},
        {
            "_id": 0,
            "name": 1,
            "description": 1,
            "detail": 1,
            "price": 1,
            "link": 1,
            "imageUrl": 1,
            "csv": 1,
            "category": 1
        }
    ).limit(candidate_limit))
    print(f"[DEBUG] 전체 제품 수 (최대 {candidate_limit}개): {len(all_products)} (DB 조회 소요시간: {time.time() - db_query_start:.2f}초)")

    # --- 가격 필터 적용 ---
    filtered_products = []
    for p in all_products:
        try:
            price = int(str(p.get("price", "0")).replace(",", ""))
            if (min_price is not None and price < min_price) or (max_price is not None and price > max_price):
                continue
        except Exception:
            continue
        filtered_products.append(p)
    print("[DEBUG] 가격 필터 후 카테고리 분포:", Counter([p.get("category", "없음") for p in filtered_products]))

    # --- 키워드 추출 및 카테고리 추론 (키워드 모듈에서 불러옴) ---
    extracted_keywords = extract_keywords_from_query(query)
    all_categories = list(set(p.get("category", "") for p in filtered_products if p.get("category")))
    category = guess_category_from_keywords(extracted_keywords, all_categories)
    print(f"[DEBUG] 추론된 카테고리: {category}")

    # --- 제품 필터링: 카테고리 필터 및 키워드 필터 적용 ---
    category_filtered = filter_products_by_category(filtered_products, category)
    print(f"[DEBUG] 카테고리 필터 후: {len(category_filtered)}")
    keyword_filtered = filter_by_query_keywords(filtered_products, query)
    print(f"[DEBUG] 키워드 필터 후: {len(keyword_filtered)}")

    # --- 후보 선택: 카테고리 필터 결과 우선, 없으면 키워드 필터, 최종적으로 전체 일부 사용 ---
    if len(category_filtered) >= 5:
        candidates = category_filtered
    elif len(keyword_filtered) >= 5:
        print("[WARN] 카테고리 제품 부족 → 키워드 필터 사용")
        candidates = keyword_filtered
    else:
        print("[WARN] 후보가 너무 적음 → 전체 일부 사용")
        candidates = filtered_products[:50]

    if not candidates:
        return [{"이름": "추천 실패", "추천이유": "조건에 맞는 제품이 없습니다."}]
    print(f"[DEBUG] 후보 선택 완료 - 후보 수: {len(candidates)}")

    # --- gemini 재랭킹: 후보 제품 수를 후보군의 상위 100개로 제한하여 프롬프트 길이를 단축 ---
    rerank_start = time.time()
    rerank_result = await rerank_ai_recommendations_async(
        room_style,
        query,
        candidates[:30],
        min_price=min_price,
        max_price=max_price,
        keyword=keyword,
        style=style,
        category=category
    )
    print(f"[DEBUG] Gemini 재랭킹 전체 소요시간: {time.time() - rerank_start:.2f}초")
    try:
        start = time.time()
        parsed = json.loads(extract_json_from_markdown(rerank_result))
        print(f"[DEBUG] 후처리 소요시간: {time.time() - start:.2f}초")
    except json.JSONDecodeError as e:
        print("JSON 파싱 실패:", e)
        print("Gemini 재랭킹 응답:\n", rerank_result)
        raise e

    print(f"[DEBUG] 최종 추천 개수: {len(parsed)}")
    print(f"[DEBUG] 전체 추천 처리 시간: {time.time() - overall_start:.2f}초")
    return parsed[:3]
