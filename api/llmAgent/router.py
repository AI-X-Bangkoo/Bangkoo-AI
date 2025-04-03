from fastapi import APIRouter, UploadFile, Form, File
from api.llmAgent.llm_agent_gimini import recommend_with_ai_agent

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

