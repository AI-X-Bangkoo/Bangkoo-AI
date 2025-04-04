from fastapi import APIRouter, UploadFile, Form, File
from api.llmAgent.llm_agent_gimini import recommend_with_ai_agent

"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

AI 추천 라우터 모듈 (FastAPI)

- /recommend 엔드포인트 정의
- 사용자로부터 이미지와 텍스트 쿼리를 받아 AI 추천 실행
"""


router = APIRouter()

@router.post("/recommend")
async def recommend(
    image: UploadFile = File(...),
    query: str = Form(...),
    min_price: int = Form(None),
    max_price: int = Form(None),
    keyword: str = Form(None),
    style: str = Form(None)
):
    result = await recommend_with_ai_agent(
        image,
        query,
        min_price=min_price,
        max_price=max_price,
        keyword=keyword,
        style=style
    )
    return result

