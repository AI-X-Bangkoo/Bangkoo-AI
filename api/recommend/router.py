from fastapi import APIRouter, Query
from typing import List, Optional
from api.recommend.style_recommender import style_recommender

router = APIRouter()

@router.get("/style-recommend")
async def style_recommend(
    styles: List[str] = Query(..., description="선호 스타일 목록 (예: 미니멀리즘, 북유럽 등)"),
    min_price: Optional[int] = Query(None),
    max_price: Optional[int] = Query(None)
):
    return await style_recommender(styles, min_price, max_price)