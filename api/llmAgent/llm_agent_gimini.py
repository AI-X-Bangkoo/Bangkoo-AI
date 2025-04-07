import json
import base64
import google.generativeai as genai
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
import os
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from utils.markdown_utils import extract_json_from_markdown
from collections import Counter

"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

gemini 기반 AI 추천 모듈

- Semantic Search로 유사 제품 후보 추출
- gemini로 재랭킹하여 최종 추천 결과 생성
"""

load_dotenv()

# Gemini 모델 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# MongoDB 연결
mongo_client = MongoClient(
    os.getenv("MONGO_URI"),
    serverSelectionTimeoutMS=30000,
    socketTimeoutMS=30000,
    connectTimeoutMS=30000
)

db = mongo_client.get_database("bangkoo")
product_collection = db.get_collection("products")


def get_room_style_description(image_path):
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

    response = model.generate_content([
        prompt,
        {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
    ])
    return response.text.strip()


def extract_keywords_from_query(query):
    return [w for w in re.findall(r'[가-힣]{2,}', query) if w not in ["추천해줘", "보여줘", "같은"]]


def guess_category_from_keywords(keywords, categories):
    for kw in reversed(keywords):
        for cat in categories:
            if kw in cat:
                return cat
    return "가구"


def filter_products_by_category(products, category: str):
    category = category.strip().lower()
    matched = []
    for p in products:
        cat = p.get("category", "").strip().lower()
        if category in cat:
            matched.append(p)
    print(f"[DEBUG] 필터링된 카테고리: {category}, 실제 매칭된 개수: {len(matched)}", flush=True)
    return matched



def filter_by_query_keywords(products, query):
    keywords = extract_keywords_from_query(query)
    print(f"[DEBUG] 추출된 키워드: {keywords}")
    filtered = []
    for p in products:
        searchable_text = f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}"
        if any(k in searchable_text for k in keywords):
            filtered.append(p)
    return filtered


def rerank_ai_recommendations(
    room_style,
    query,
    candidate_products,
    previous_results=None,
    min_price=None,
    max_price=None,
    keyword=None,
    style=None,
    category=None
):
    product_descriptions = "\n\n".join([
        f"""
        이름: {p['name']}
        설명: {p['description']}
        상세설명: {p.get('detail', '정보 없음')}
        할인가: {p.get('price', '정보 없음')}
        정상가: {p.get('price', '정보 없음')}
        링크: {p['link']}
        이미지: {p['imageUrl']}
        """
        for p in candidate_products
    ])

    filter_info = ""
    if min_price is not None and max_price is not None:
        filter_info += f"- 가격대: {min_price:,} ~ {max_price:,}원\n"
    if keyword:
        filter_info += f"- 키워드 포함: {keyword}\n"
    if style:
        filter_info += f"- 원하는 스타일: {style}\n"
    if category:
        filter_info += f"- 제품 종류 (카테고리): {category}\n"

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
        prompt += f"""

[이전 추천 결과]
{prev_json}

위 결과는 사용자의 취향과 약간 다를 수 있습니다. 새 요청을 참고해 다시 3개의 제품을 추천해 주세요.
"""

    if category:
        prompt += f"""
- 반드시 '{category}'에 해당하는 제품만 추천해 주세요.
- '{category}'가 아닌 제품은 추천 이유에 '해당 카테고리가 아니므로 제외'라고 써 주세요.
- 아래 제품 목록 외에는 절대 새로운 제품을 만들어내지 마세요.
- 링크/이미지/이름 등을 임의로 작성하지 마세요. 목록에 있는 제품만 그대로 추천해 주세요.
"""

    prompt += f"""

[추천 후보 제품 목록]
아래 제품들은 실제 데이터베이스에서 검색된 후보입니다.
이 중에서 방 스타일과 사용자 요청, 필터 조건에 가장 적합한 3개 이상의 제품을 골라 주세요.
- 가격대({min_price} ~ {max_price})를 벗어나는 제품은 추천하지 마세요.
- 반드시 3개 이상의 제품을 JSON 배열로 반환하세요. 조건 미충족 시 "추천이유"에 이유를 써 주세요

{product_descriptions}

결과는 아래 JSON 형식으로만 반환하세요:
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
    response = model.generate_content(prompt)
    print("[DEBUG] Gemini 응답:\n", response.text)
    return response.text.strip()


async def recommend_with_ai_agent(
    image_file: UploadFile,
    query: str,
    min_price: int = None,
    max_price: int = None,
    keyword: str = None,
    style: str = None
):
    temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(await image_file.read())
    temp_file.close()

    room_style = get_room_style_description(temp_file.name)

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
    ).limit(1000))

    print(f"[DEBUG] 전체 제품 수: {len(all_products)}")

    filtered_products = []
    for p in all_products:
        try:
            price = int(str(p.get("price", "0")).replace(",", ""))
            if (min_price is not None and price < min_price) or (max_price is not None and price > max_price):
                continue
        except:
            continue
        filtered_products.append(p)
    print("[DEBUG] 카테고리 분포 (가격 필터 후):", flush=True)
    print(Counter([p.get("category", "없음") for p in filtered_products]), flush=True)

    keywords = extract_keywords_from_query(query)

    # 실존하는 카테고리 중 가장 잘 맞는 것 추정
    all_categories = list(set(p.get("category", "") for p in filtered_products if p.get("category")))
    category = guess_category_from_keywords(keywords, all_categories)
    print(f"[DEBUG] 추론된 카테고리: {category}")

    category_filtered = filter_products_by_category(filtered_products, category)
    print(f"[DEBUG] 카테고리 필터 후: {len(category_filtered)}")

    keyword_filtered = filter_by_query_keywords(filtered_products, query)
    print(f"[DEBUG] 키워드 필터 후: {len(keyword_filtered)}")

    # 후보 선택 로직 개선
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

    result = rerank_ai_recommendations(
        room_style,
        query,
        candidates[:50],  # ⬅ Gemini로 넘기는 후보 수 확대
        min_price=min_price,
        max_price=max_price,
        keyword=keyword,
        style=style,
        category=category
    )
    try:
        parsed = json.loads(extract_json_from_markdown(result))
    except json.JSONDecodeError as e:
        print("JSON 파싱 실패:", e)
        print("Gemini 응답:\n", result)
        raise e

    print(f"[DEBUG] 최종 추천 개수: {len(parsed)}")
    return parsed[:3]
