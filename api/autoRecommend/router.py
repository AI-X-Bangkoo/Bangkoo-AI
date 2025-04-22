from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from api.recommend import recommend_furniture_for_room  # 추천 함수

router = APIRouter()

class RoomImageRequest(BaseModel):
    room_image: str  # 방 이미지를 base64로 전달하거나 URL을 사용할 수 있습니다
    style_keywords: List[str]  # 스타일에 관련된 키워드 리스트
    min_price: int = None  # 최소 가격 필터
    max_price: int = None  # 최대 가격 필터

class RecommendedProduct(BaseModel):
    이름: str
    설명: str
    가격: str
    링크: str
    이미지: str
    추천이유: str

@router.post("/style-recommendation", response_model=List[RecommendedProduct])
async def get_style_recommendation(request: RoomImageRequest):
    """
    방 이미지 및 스타일 키워드를 기반으로 적합한 가구를 추천합니다.
    """
    try:
        room_image = request.room_image
        style_keywords = request.style_keywords
        min_price = request.min_price
        max_price = request.max_price

        recommended_products = await recommend_furniture_for_room(
            room_image, style_keywords, min_price, max_price
        )
        
        return recommended_products
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 오류: {str(e)}")
