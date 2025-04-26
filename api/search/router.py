from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
import requests, tempfile, io
from typing import Optional
from api.llmAgent.llm_agent_gimini import recommend_with_ai_agent
from api.search.image_search import image_search
from api.search.hybrid_search import hybrid_search
from utils.extract_direct_image_url import extract_direct_image_url
from utils.gemini_utils import should_use_image_for_recommendation
import base64
import re
from utils.query_utils import is_valid_query

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
    user_id: Optional[str] = Form(None)
):
    # print("[DEBUG] /search 진입")
    try:
        form_data = await request.form()
        # print("전체 요청 form 데이터:", dict(form_data))
    except Exception as e:
        raise ValueError("form_data 추출 중 오류 발생:", e)

    # print("query:", query)
    # print("image:", image)
    # print("image_url:", image_url)

    contents = None
    image_upload_file = None

    # 파일 업로드가 있으면 먼저 읽어서 contents에 저장하고, 새 UploadFile 생성
    if image:
        contents = await image.read()
        # print("[DEBUG] 업로드된 이미지 파일 사용")
        # 새 UploadFile 생성: BytesIO에 contents를 담아 전달
        image_upload_file = StarletteUploadFile(filename=image.filename, file=io.BytesIO(contents))
    # 파일 업로드가 없고 image_url이 있으면 image_url 처리
    elif image_url:
        try:
            if image_url.startswith("http"):
                # 직접 접근 가능한 이미지 URL 처리
                true_url = extract_direct_image_url(image_url)
                # print(f"[DEBUG] 이미지 URL 변환: {true_url}")
                response = requests.get(true_url)
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="이미지 URL 접근 실패")
                contents = response.content
            else:
                # base64 처리
                # print("[DEBUG] image_url 파라미터가 base64 데이터인 것으로 간주")
                base64_data = re.sub("^data:image/.+;base64,", "", image_url)
                contents = base64.b64decode(base64_data)

            if not contents:
                raise HTTPException(status_code=400, detail="이미지 데이터를 불러오지 못했습니다.")

            # UploadFile 객체 생성
            image_upload_file = StarletteUploadFile(filename="temp.jpg", file=io.BytesIO(contents))

        except Exception as e:
            # print(f"[ERROR] image_url 처리 실패: {e}")
            raise HTTPException(status_code=400, detail="image_url 처리 실패")


    # 이미지 단독 검색: 쿼리가 없으면 이미지 기반 검색 실행
    if contents is not None and not is_valid_query(query):
        # print("[DEBUG] 이미지 단독 검색으로 분기")
        return image_search(contents)

    # 이미지 + 쿼리 (추천 요청)
    if contents is not None and is_valid_query(query):
        # print("[DEBUG] 이미지 + 쿼리 기반 분기 → 무조건 추천")
        return await recommend_with_ai_agent(
            image_upload_file,  # 업로드된 이미지
            query
        )

    # 쿼리만 있는 경우 (이미지 없이)
    if is_valid_query(query):
        # print("[DEBUG] 텍스트 하이브리드 검색으로 분기")
        return hybrid_search(query)

    raise HTTPException(status_code=400, detail="유효한 검색 조건이 없습니다.")