from fastapi import APIRouter, UploadFile, File, Form
from .img_generation_gemini import process_placement

router = APIRouter()

@router.post("/placement")  # ← 공통 엔드포인트
async def generate_image(
        mode: str = Form(...),  # 👈 작업 타입 (add / remove 등)
        background: UploadFile = File(...),
        reference: UploadFile = File(None)  # add일 때만 필요
):
    result = await process_placement(mode, background, reference)
    return result
