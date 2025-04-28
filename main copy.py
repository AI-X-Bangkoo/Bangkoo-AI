from fastapi import FastAPI
from api.search.router import router as search_router
from api.placement.router import router as placement_router
# from api.detection.sam2_dino_mask_detection_router import router as sam2_dino_mask_detection_router
from model_loader import model_manager
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from utils.constants import UPLOAD_DIR
from api.recommend.router import router as style_recommend_router
import threading
import asyncio
from utils.query_utils import load_keyword_cache
from mongo_manager import mongo_manager
from api.productsEmbedding.router import router as embedding_router
from api.autoRecommend.router import router as style_recommendation
from api.autoRecommend.router import router as analyz_room

app = FastAPI()

"""
최초 작성자: 김동규
최초 작성일: 2025-04-07
설명: FastAPI 애플리케이션을 구성하고, 모델 및 MongoDB를 startup 이벤트에서 초기화하며, 각 기능별 API 라우터를 등록
"""

# 2025-04-12 김범석 추가 (static 파일 접근)
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    print("startup_event 시작")
    load_keyword_cache(mongo_manager.db)
    async def async_model_load():
        try:
            await asyncio.to_thread(model_manager.load)
        except Exception as e:
            print("모델 로딩 중 에러 발생:", e)


# await을 직접 사용해서 모델 로딩이 끝날 때까지 정확히 기다림
# 그 전에 어떤 API 요청도 못 받음 (FastAPI가 기다려줌) -김병훈 수정정
    await asyncio.to_thread(model_manager.load)
    print("startup_event 끝")

app.include_router(search_router, prefix="/api")
app.include_router(placement_router, prefix="/api")
app.include_router(style_recommend_router, prefix="/api")
app.include_router(style_recommendation, prefix="/api")
app.include_router(analyz_room, prefix="/api")
app.include_router(embedding_router, prefix="/api")
# app.include_router()


@app.get("/")
def read_root():
    return {"status": "ok", "model_ready": model_manager.ready}

# ASGI application reference
a00 = app