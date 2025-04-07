from fastapi import APIRouter, UploadFile, Form, File, HTTPException
from typing import Optional
from api.llmAgent.llm_agent_gimini import recommend_with_ai_agent
from api.search.image_search import image_search
from api.search.hybrid_search import hybrid_search

router = APIRouter()

@router.post("/recommend-or-search")
async def recommend_or_search(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    min_price: Optional[int] = Form(None),
    max_price: Optional[int] = Form(None),
    keyword: Optional[str] = Form(None),
    style: Optional[str] = Form(None)
):
    print("[DEBUG] /recommend-or-search 진입")
    print("query:", query)
    print("image:", image)
    print("url:", url)

    query_lower = query.lower()

    # Gemini 기반 추천
    if "추천" in query_lower or "어울리" in query_lower:
        if image is None:
            raise HTTPException(status_code=400, detail="추천 요청은 이미지 파일이 필요합니다.")
        return await recommend_with_ai_agent(
            image,
            query,
            min_price=min_price,
            max_price=max_price,
            keyword=keyword,
            style=style
        )

    # 이미지 유사도 검색
    elif "비슷" in query_lower or "같은" in query_lower:
        if image:
            contents = await image.read()
        elif url:
            import requests
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="이미지 URL 접근 실패")
            contents = response.content
        else:
            raise HTTPException(status_code=400, detail="이미지 또는 URL이 필요합니다.")

        return image_search(contents)

    # 텍스트 기반 하이브리드 검색
    elif query:
        return hybrid_search(query)

    else:
        raise HTTPException(status_code=400, detail="유효한 검색 조건이 없습니다.")
