from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
import requests, tempfile, io
from typing import Optional
from api.llmAgent.llm_agent_gimini import recommend_with_ai_agent
from api.search.image_search import image_search
from api.search.hybrid_search import hybrid_search
from utils.extract_direct_image_url import extract_direct_image_url
from utils.gemini_utils import should_use_image_for_recommendation


"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

하이브리드 검색 라우터 (FastAPI)

- /search 엔드포인트 정의
- 사용자 쿼리를 기반으로 AI 검색 수행
"""

router = APIRouter()

@router.post("/search")
async def recommend_or_search(
    request: Request,
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    min_price: Optional[int] = Form(None),
    max_price: Optional[int] = Form(None),
    keyword: Optional[str] = Form(None),
    style: Optional[str] = Form(None)
):
    print("[DEBUG] /search 진입")

    try:
        form_data = await request.form()
        print("전체 요청 form 데이터:", dict(form_data))
    except Exception as e:
        print("form_data 추출 중 오류 발생:", e)

    query_lower = (query or "").lower()
    print("query:", query)
    print("image:", image)
    print("image_url:", image_url)

    contents = None

    # 1. 이미지 URL만 있는 경우 → 다운로드
    if image is None and image_url:
        true_url = extract_direct_image_url(image_url)
        print(f"[DEBUG] 이미지 URL 변환: {true_url}")

        try:
            response = requests.get(true_url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="이미지 URL 접근 실패")
            contents = response.content
        except Exception as e:
            print(f"[ERROR] 이미지 다운로드 실패: {e}")
            raise HTTPException(status_code=400, detail="이미지 다운로드 실패")

    # 2. 업로드된 이미지 파일인 경우
    if image:
        contents = await image.read()

    # 이미지만 존재 → 이미지 유사도 검색
    if contents is not None and (query is None or query.strip() == ""):
        print("[DEBUG] 이미지 단독 검색으로 분기")
        return image_search(contents)

    # 이미지 + 쿼리 (추천 요청) → Gemini
    if contents is not None and query:
        print("[DEBUG] 이미지 + 쿼리 기반 분기 시작")

        if should_use_image_for_recommendation(query):
            print("[DEBUG] Gemini 판단 결과: 이미지 기반 추천 필요 → Gemini 추천으로 분기")
            image_upload_file = image or StarletteUploadFile(filename="temp.jpg", file=io.BytesIO(contents))
            return await recommend_with_ai_agent(
                image_upload_file,
                query,
                min_price=min_price,
                max_price=max_price,
                keyword=keyword,
                style=style
            )
        else:
            print("[DEBUG] Gemini 판단 결과: 이미지 사용 안함 → 텍스트 기반 하이브리드 검색")


    # 텍스트 기반 하이브리드 검색
    if query:
        print("[DEBUG] 텍스트 하이브리드 검색으로 분기")
        return hybrid_search(query)

    raise HTTPException(status_code=400, detail="유효한 검색 조건이 없습니다.")

