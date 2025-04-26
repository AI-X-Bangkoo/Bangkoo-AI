import json
import os
import re
import sys
import time
import asyncio
from tempfile import NamedTemporaryFile, gettempdir
from fastapi import UploadFile
from dotenv import load_dotenv
import google.generativeai as genai
from utils.markdown_utils import extract_json_from_markdown
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
최초 작성일: 2025-04-20
수정일: 2025-04-26 (김동규) (전체 추천 파이프라인 흐름 주석 추가)

Gemini 기반 AI 추천 모듈 (Recommend with AI Agent)

[전처리 단계]
1. 업로드된 방 이미지 임시 저장
2. Gemini 2.0-flash 모델을 사용해 방 스타일 설명(room style description) 생성
   - 이미지에 대해 2~3문장 객관적 스타일 설명 생성
   - 생성 결과를 캐시 디렉토리에 저장하여 재사용 최적화

[검색 단계]
3. MongoDB에서 전체 제품 목록 조회
   - 기본 필드(name, description, detail, price, link, imageUrl, category)만 선택
4. 입력 쿼리 분석 및 후보 제품 필터링
   - 쿼리에서 키워드 추출 → 카테고리 추정
   - 카테고리 기반 후보 필터링 (부족하면 키워드 기반, 최종 fallback은 전체)
   - 최대 10개까지 후보 제품 선택

[후처리 단계]
5. 후보 제품에 대해 Gemini 2.0-flash를 호출하여 통합 재랭킹 및 추천 이유 생성
   - 방 스타일, 쿼리, 후보 제품 요약을 하나의 프롬프트로 전달
   - 최적의 3개 이상 제품을 선정하고 추천 이유(reason)를 JSON 배열로 반환
6. 추천 결과 매칭 및 포맷 변환
   - Gemini 응답의 제품명을 기존 후보 리스트와 매칭하여 최종 결과 조합
   - 제품 이름, 설명, 링크, 이미지, 할인가, 추천 이유를 포함하여 반환

[특징 및 참고]
- 방 스타일 설명은 최초 요청 시 Gemini 호출, 이후 캐시 활용
- 후보 선정 로직은 카테고리 → 키워드 → 전체 순으로 fallback
- 재랭킹과 추천 이유 생성을 단일 Gemini 호출로 처리하여 통합 최적화
- JSON 파싱 실패 대비를 위해 markdown block에서 안전 추출
"""

# 환경 변수 및 Gemini 설정
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.0-flash")

# MongoDB 연결 보장
if not mongo_manager.ready:
    mongo_manager.connect()
product_collection = mongo_manager.products

# 캐시 디렉토리 설정
CACHE_DIR = os.path.join(gettempdir(), "room_style_cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cached_room_style(image_path: str) -> str:
    cache_path = os.path.join(CACHE_DIR, f"room_style_{os.path.basename(image_path)}.txt")
    if os.path.exists(cache_path):
        return open(cache_path, "r", encoding="utf-8").read()
    return None

def cache_room_style(image_path: str, description: str) -> None:
    cache_path = os.path.join(CACHE_DIR, f"room_style_{os.path.basename(image_path)}.txt")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(description)

async def get_room_style_description_async(image_path: str) -> str:
    """방 스타일 설명 생성 및 캐시"""
    start = time.time()
    cached = get_cached_room_style(image_path)
    if cached:
        # print(f"[DEBUG] 캐시에서 방 스타일 불러옴 ({time.time()-start:.2f}s)")
        return cached
    prompt = """
아래 방 사진을 보고 객관적인 스타일 설명을 2~3문장으로 작성해 주세요.
"""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    resp = await asyncio.to_thread(model.generate_content, [prompt, {"mime_type":"image/jpeg","data":img_bytes}])
    desc = resp.text.strip()
    cache_room_style(image_path, desc)
    # print(f"[DEBUG] Gemini 방 스타일 생성 ({time.time()-start:.2f}s)")
    return desc

async def rerank_ai_recommendations_async(
    room_style: str,
    query: str,
    candidate_products: list,
    previous_results=None,
    category=None
) -> str:
    """후보 3개 이상 선택 + 추천 이유를 단일 프롬프트로 생성"""
    # 각 후보 제품 요약
    summary = "\n".join(
        f"이름: {p['name']}, 가격: {p.get('price','-')}, 링크: {p['link']}, 설명: {p.get('description','')}"
        for p in candidate_products
    )
    # 통합 프롬프트
    prompt = f"""
당신은 인테리어 전문가입니다.
[방 스타일]
{room_style}
[요청]
{query}
"""
    if previous_results:
        prev_json = json.dumps(previous_results, ensure_ascii=False, indent=2)
        prompt += f"\n[이전 추천 결과]\n{prev_json}\n위 결과를 참고하여 다시 3개 이상의 제품을 추천해 주세요."
    if category:
        prompt += f"\n- 반드시 '{category}' 카테고리 제품만 추천하고, 다른 제품은 제외 이유를 적어주세요."
    prompt += f"""
[후보 제품]
{summary}
위 목록에서 가장 적합한 3개 이상 제품을 선택하고, 한 문장 추천 이유와 함께 JSON 배열로 반환하세요.
형식:
[
  {{"name":"...","reason":"...","link":"...","image":"...","price":"..."}},
  ...
]
"""
    start = time.time()
    resp = await asyncio.to_thread(model.generate_content, [prompt])
    # print(f"[DEBUG] 통합 재랭킹+이유 생성 ({time.time()-start:.2f}s)")
    return resp.text.strip()

async def recommend_with_ai_agent(
    image_file: UploadFile,
    query: str,
    category: str = None
) -> list:
    """메인 추천 함수"""
    overall_start = time.time()
    # print(f"[DEBUG] /search 진입: {query}")

    # 1) 이미지 임시 저장
    tmp = NamedTemporaryFile(delete=False, suffix='.jpg')
    tmp.write(await image_file.read())
    tmp.close()

    # 2) 방 스타일 생성
    room_style = await get_room_style_description_async(tmp.name)

    # 3) DB 후보 조회 및 필터링
    if not mongo_manager.ready:
        mongo_manager.connect()
    all_docs = list(product_collection.find({}, {"_id":0, "name":1, "description":1, "detail":1, "price":1, "link":1, "imageUrl":1, "category":1}))
    # 가격, 키워드, 스타일 필터 제거: 카테고리만 사용
    extracted = extract_keywords_from_query(query)
    cat = guess_category_from_keywords(extracted, [p.get('category','') for p in all_docs])
    cat_filtered = filter_products_by_category(all_docs, cat)
    kw_filtered = filter_by_query_keywords(all_docs, query)
    candidates = cat_filtered if len(cat_filtered) >= 5 else (kw_filtered if len(kw_filtered) >= 5 else all_docs)
    candidates = candidates[:10]
    # print(f"[DEBUG] 후보 수: {len(candidates)} (category filter 적용)")

    # 4) 통합 재랭킹 + 추천이유 생성
    raw = await rerank_ai_recommendations_async(
        room_style=room_style,
        query=query,
        candidate_products=candidates,
        previous_results=None,
        category=cat
    )
    items = json.loads(extract_json_from_markdown(raw))

        # 5) 결과 조합 및 이름 매칭 핸들링
    result = []
    
    # 이름 매칭 헬퍼 함수
    def find_candidate(name):
        # 정확 일치 우선, 부분 문자열 매칭으로 보조
        for p in candidates:
            if p['name'] == name:
                return p
        for p in candidates:
            if name in p['name'] or p['name'] in name:
                return p
        return None

    for o in items:
        cand = find_candidate(o.get('name', ''))
        if not cand:
            # print(f"[WARNING] 매칭되는 후보 없음: {o.get('name')}")
            continue
        # 'reason' 또는 '추천이유' 키 지원
        reason = o.get('reason') or o.get('추천이유') or ''
        result.append({
            "이름":     cand['name'],
            "설명":     cand.get('description', ''),
            "링크":     cand['link'],
            "이미지":   cand.get('imageUrl', ''),
            "상세설명": cand.get('detail', ''),
            "할인가":   cand.get('price'),
            "추천이유": reason
        })

    print(f"[DEBUG] 전체 처리 시간: {time.time() - overall_start:.2f}s")
    return result
