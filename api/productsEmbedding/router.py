from fastapi import APIRouter, HTTPException, File, UploadFile
from io import StringIO
import csv
from typing import List
from api.productsEmbedding.upload_products_embedding import upload_products  # 임베딩 함수
from datetime import datetime
from pydantic import BaseModel
import traceback
from typing import Optional

# FastAPI 라우터 정의
router = APIRouter()

# Pydantic 스키마 정의
class Product(BaseModel):
    _id: str  # MongoDB의 _id 필드를 포함
    name: str
    description: Optional[str] = ""
    detail: Optional[str] = ""
    imageUrl: Optional[str] = None
    link: str

# CSV 파일에서 제품 데이터 추출
def parse_csv_file(file: UploadFile) -> List[Product]:
    try:
        # CSV 파일을 문자열로 읽기
        content = file.file.read().decode("utf-8")
        csv_reader = csv.DictReader(StringIO(content))
        
        products = []
        for row in csv_reader:
            product = Product(
                _id=row["_id"],
                name=row["name"],
                description=row.get("description", ""),
                detail=row.get("detail", ""),
                imageUrl=row.get("imageUrl", None),
                link=row["link"]
            )
            products.append(product)
        
        return products
    
    except Exception as e:
        print(f"[에러] CSV 파일 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=400, detail="CSV 파일을 처리하는 중 오류가 발생했습니다.")

# CSV 파일 업로드 및 임베딩 처리 API
@router.post("/admin/CSVupload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # CSV 파일 파싱
        products = parse_csv_file(file)

        # 임베딩 처리 함수 호출
        result = upload_products([p.dict() for p in products])

        # 임베딩 완료 후 응답
        return {"message": "CSV 파일 업로드 및 임베딩 완료", "갯수": len(products), "결과": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CSV 파일 업로드 및 처리 중 오류 발생: {e}")

