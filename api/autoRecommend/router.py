# api/autoRecommend/router.py
from fastapi import APIRouter, UploadFile, HTTPException, File
from typing import List
from pydantic import BaseModel
# 함수만 정확히 import
from api.autoRecommend.recommend_furniture_for_room import recommend_furniture_for_room

router = APIRouter()

class RecommendedProduct(BaseModel):
    이름: str
    설명: str
    링크: str
    이미지: str
    가격: str
    추천이유: str

@router.post("/analyze_room", response_model=List[RecommendedProduct])
async def analyze_and_recommend(
    file: UploadFile = File(...),
    style_keywords: List[str] = None,
    min_price: int = None,
    max_price: int = None
): 
    try:
        # 추천 함수 호출
        recommended_products = await recommend_furniture_for_room(
            file, style_keywords, min_price, max_price
        )
        return recommended_products

    except Exception as e:
        print("❌ 분석 및 추천 파이프라인 중 오류 발생:", str(e))
        raise HTTPException(status_code=500, detail=f"분석 및 추천 오류: {str(e)}")