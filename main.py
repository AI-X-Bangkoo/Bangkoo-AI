from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import asyncio

from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.constants import UPLOAD_DIR
from utils.query_utils import load_keyword_cache

# Routers
from api.search.router import router as search_router
from api.placement.router import router as placement_router
from api.recommend.router import router as style_recommend_router
from api.productsEmbedding.router import router as embedding_router
from api.autoRecommend.router import router as auto_recommend_router
from api.detection.router import router as detection_router

# Vector index
from api.autoRecommend.vector_index import vector_index

# --- 2. FastAPI 객체 생성 ---
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
    print("▶ startup_event 시작")
    load_keyword_cache(mongo_manager.db)
    print("✔ Keyword cache loaded")
    try:
        await asyncio.to_thread(model_manager.load)
        print("✔ Model loaded")
    except Exception as e:
        print("❗ Model loading error:", e)
    try:
        await asyncio.to_thread(vector_index.build)
        print(f"✔ Faiss index built: total docs = {len(vector_index.product_docs)}")
    except Exception as e:
        print("❗ Vector index build error:", e)
    print("▶ startup_event 완료")

# ---  Routers 등록 ---
app.include_router(search_router, prefix="/api")
app.include_router(placement_router, prefix="/api")
app.include_router(style_recommend_router, prefix="/api")
app.include_router(auto_recommend_router, prefix="/api")
app.include_router(embedding_router, prefix="/api")
app.include_router(detection_router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok", "model_ready": model_manager.ready}

# --- ASGI Reference ---
a00 = app
