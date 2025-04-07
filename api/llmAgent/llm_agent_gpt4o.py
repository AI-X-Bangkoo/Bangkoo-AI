import json
import base64
import openai
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
from dotenv import load_dotenv
from api.search.hybrid_search import hybrid_search
from api.search.filters import apply_filters
import os
from utils.markdown_utils import extract_json_from_markdown

"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

GPT-4o 기반 AI 추천 모듈

- 이미지(base64 인코딩)를 GPT-4o에 전달하여 방 스타일 설명 생성
- Semantic Search로 유사 제품 후보 추출
- GPT-4o로 재랭킹하여 최종 추천 결과 생성
"""


# 환경변수 로드 및 클라이언트 설정
load_dotenv()
oai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. 방 스타일 설명 생성
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
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# 2. 재랭킹 추천 생성
def rerank_ai_recommendations(
    room_style,
    query,
    candidate_products,
    previous_results=None,
    min_price=None,
    max_price=None,
    keyword=None,
    style=None
):
    product_descriptions = "\n\n".join([
        f"""
        이름: {p['이름']}
        설명: {p['설명']}
        상세설명: {p.get('상세설명', '정보 없음')}
        할인가: {p.get('할인가', '정보 없음')}
        정상가: {p.get('정상가', '정보 없음')}
        링크: {p['링크']}
        이미지: {p['이미지']}
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

    prompt += f"""

[추천 후보 제품 목록]
아래 제품들은 Semantic Search를 통해 선별된 후보입니다.
이 중에서 방 스타일, 사용자 요청, 필터 조건에 가장 적합한 3개 제품을 골라 주세요.

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

    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# 3. FastAPI용 통합 함수
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
    semantic_query = f"{room_style} {query}"
    candidates = hybrid_search(semantic_query, top_k=50)

    price_range = (min_price, max_price) if min_price is not None and max_price is not None else None
    filtered_candidates = apply_filters(candidates, price_range, keyword, style)

    if len(filtered_candidates) < 3:
        filtered_candidates = candidates[:10]

    result = rerank_ai_recommendations(
        room_style,
        query,
        filtered_candidates[:10],
        min_price=min_price,
        max_price=max_price,
        keyword=keyword,
        style=style
    )

    try:
        parsed = json.loads(extract_json_from_markdown(result))
    except json.JSONDecodeError as e:
        print("JSON 파싱 실패:", e)
        print("Gemini 응답:\n", result)
        raise e

    print(f"[DEBUG] 최종 추천 개수: {len(parsed)}")
    return parsed[:3]
