from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from api.autoRecommend import recommend_furniture_for_room  # 추천 함수
from utils.gemini_utils import analyze_room_with_gemini_by_file

router = APIRouter()

# RecommendedProduct 모델 정의
class RecommendedProduct(BaseModel):
    이름: str
    설명: str
    가격: str
    링크: str
    이미지: str
    추천이유: str

print(f"[DEBUG] analyze_room_with_gemini_by_file 타입: {(analyze_room_with_gemini_by_file)}")

# 스타일 추천 API (이미지 파일 업로드 방식)
@router.post("/style_recommendation", response_model=List[RecommendedProduct])
async def get_style_recommendation(
    file: UploadFile = File(...),
    style_keywords: List[str] = None,
    min_price: int = None,
    max_price: int = None,
):
    """
    업로드된 방 이미지 및 스타일 키워드를 기반으로 적합한 가구를 추천합니다.
    """
    try:
        recommended_products = await recommend_furniture_for_room(
            file, style_keywords, min_price, max_price
        )
        return recommended_products
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 오류: {str(e)}")


# 방 스타일 분석 API
@router.post("/analyze-room")
async def analyze_room(file: UploadFile = File(...)):  # 이미지 파일 받기
    """
    업로드된 방 이미지를 분석하여 스타일, 가구 카테고리 등을 추출합니다.
    """
    try:
        # 이미지를 분석하는 함수 호출 (파일 처리)
        result = await analyze_room_with_gemini_by_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")