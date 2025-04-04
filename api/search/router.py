from fastapi import APIRouter, Form
from api.search.hybrid_search import hybrid_search
from api.search.filters import apply_filters

"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

하이브리드 검색 라우터 (FastAPI)

- /search 엔드포인트 정의
- 사용자 쿼리를 기반으로 AI 검색 수행
"""


router = APIRouter()

@router.post("/search")
def search(
    query: str = Form(...),
    min_price: int = Form(None),
    max_price: int = Form(None),
    keyword: str = Form(None),
    style: str = Form(None)
):
    results = hybrid_search(query, top_k=10)

    price_range = (min_price, max_price) if min_price and max_price else None
    filtered = apply_filters(results, price_range, keyword, style)

    return filtered
