from fastapi import FastAPI
from api.llmAgent.router import router as recommend_router
from api.search.router import router as search_router

app = FastAPI()

# 라우터 등록
app.include_router(recommend_router, prefix="/ai")
app.include_router(search_router, prefix="/ai")

# 기본 헬스체크
@app.get("/")
def read_root():
    return {"status": "ok"}
