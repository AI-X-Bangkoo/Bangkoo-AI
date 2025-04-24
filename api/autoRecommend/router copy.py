from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from api.autoRecommend import recommend_furniture_for_room  # 추천 함수
from utils.image_analysis_utils import analyze_room_with_gemini_by_file
from pydantic import BaseModel 
import logging
import traceback

router = APIRouter()
logging.basicConfig(level=logging.DEBUG)

# RecommendedProduct 모델 정의
class RecommendedProduct(BaseModel):
    이름: str
    설명: str
    가격: str
    링크: str
    이미지: str
    추천이유: str

@router.post("/analyze_room")
async def post_analyze_room(file: UploadFile = File(...)):  # 이미지 파일 받기
    print(f">>>> 업로드된 파일: {file}")
    if file is None:
        raise HTTPException(status_code=400, detail="파일이 전달되지 않았습니다.")
    """
    업로드된 방 이미지를 분석하여 스타일, 가구 카테고리 등을 추출합니다.
    """
    try:
        # 이미지를 분석하는 함수 호출 (파일 처리)
        result = await analyze_room_with_gemini_by_file(file)
        print(result)
        return result
    except Exception as e:
        print("[디버그] 예외 발생: ") 
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")

# 스타일 추천 API
@router.post("/style_recommendation", response_model=List[RecommendedProduct])
async def get_style_recommendation(
    file: UploadFile = File(...)):
    """
    업로드된 방 이미지 및 스타일 키워드를 기반으로 적합한 가구를 추천합니다.
    """
    try:
        recommended_products = await recommend_furniture_for_room(file)
        print(recommended_products)
        return recommended_products
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 오류: {str(e)}")
