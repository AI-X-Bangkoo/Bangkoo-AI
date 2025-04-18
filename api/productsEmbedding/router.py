from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from api.productsEmbedding.upload_products_embedding import upload_products
import traceback


"""
최초 작성자: 김병훈
최초 작성일: 2025-04-17

product저장시 임베딩 라우터

-/embedding 엔드포인트 정의
-
"""

router = APIRouter()

#Pydantic 스키마 정의
class Product(BaseModel):
    name:str
    description: Optional[str] = ""
    detail: Optional[str] = ""
    imageUrl: Optional[str] = None
    link: str


#POST API 라우터
@router.post("/embedding")
async def embedding_router(products: List[Product]):
    try: 
        #Pydantic 객체들을 dict로 변환
        product_dicts = [p.dict() for p in products]

        #기존 업로드 함수 호출
        upload_products(product_dicts)

        return {"[알림]" : "임베딩 및 저장 완료", "갯수 :" :len(products)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code = 500, detail = f"임베딩 처리중 오류 발생: {e}")
    
