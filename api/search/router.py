from fastapi import APIRouter, Form
from api.search.hybrid_search import hybrid_search
from api.search.filters import apply_filters

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
