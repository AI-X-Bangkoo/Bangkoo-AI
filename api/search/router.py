from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from api.search.hybrid_search import hybrid_search
from api.search.filters import apply_filters
from api.search.image_search import image_search
import base64
import re
import requests


"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

하이브리드 검색 라우터 (FastAPI)

- /search 엔드포인트 정의
- 사용자 쿼리를 기반으로 AI 검색 수행
"""


router = APIRouter()

def extract_base64_image(data_uri):
    match = re.match(r"data:image/[^;]+;base64,(.+)", data_uri)
    if not match:
        raise HTTPException(status_code=400, detail="Base64 형식이 잘못되었습니다.")
    return base64.b64decode(match.group(1))

@router.post("/search")
def search(
    query: str = Form(...),
    min_price: int = Form(None),
    max_price: int = Form(None),
    keyword: str = Form(None),
    style: str = Form(None)
):
    try:
        results = hybrid_search(query, top_k=10)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")

    price_range = (min_price, max_price) if min_price and max_price else None
    filtered = apply_filters(results, price_range, keyword, style)

    return filtered

@router.post("/search/image")
async def image_based_search(
    file: UploadFile = File(None),
    url: str = Form(None),
    min_price: int = Form(None),
    max_price: int = Form(None),
    keyword: str = Form(None),
    style: str = Form(None)
):
    print("[1] 요청 수신됨")
    if file is not None:
        print("[2] file 방식 수신")
        contents = await file.read()
    elif url is not None:
        print(f"[2] url 방식 수신: {url[:50]}...")
        if url.startswith("data:image/"):
            print("[3] data URI base64 추출 중")
            contents = extract_base64_image(url)
        else:
            print("[3] 외부 이미지 요청 중")
            response = requests.get(url)
            print(f"[4] 응답 코드: {response.status_code}")
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="이미지 URL을 불러올 수 없습니다.")
            contents = response.content
    else:
        print("[2] file, url 모두 없음")
        raise HTTPException(status_code=400, detail="file 또는 url 중 하나가 필요합니다.")

    print("[5] image_search 호출 전")
    results = image_search(contents, top_k=10)
    print("[6] image_search 결과 수신 완료")
    price_range = (min_price, max_price) if min_price and max_price else None
    filtered = apply_filters(results, price_range, keyword, style)
    print("[7] 필터링 완료")
    return filtered
