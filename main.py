from fastapi import FastAPI
from api.llmAgent.router import router as recommend_router
from api.search.router import router as search_router
from api.placement.router import router as placement_router
from model_loader import model_manager
from api.llmAgent.router import router as recommend_or_search_router
import threading
import asyncio

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("startup_event 시작")

    async def async_model_load():
        try:
            await asyncio.to_thread(model_manager.load)
        except Exception as e:
            print("모델 로딩 중 에러 발생:", e)

    asyncio.create_task(async_model_load())
    print("startup_event 끝")


app.include_router(recommend_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(recommend_or_search_router, prefix="/api")
app.include_router(placement_router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok", "model_ready": model_manager.ready}
