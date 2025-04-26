import re
from collections import Counter

"""
최초 작성자: 김동규
최초 작성일: 2025-04-11

- 사용자 쿼리에서 한글 키워드 추출 및 필터링
- 키워드를 기반으로 카테고리를 유추하거나 제품을 필터링
- 제품 정보에서 name, description, detail을 통합 검색 대상으로 사용
- 불필요한 단어 필터링 ('추천해줘', '보여줘' 등)
- 카테고리 필터는 소문자 비교 기반으로 일치 여부 판별
"""

def extract_keywords_from_query(query: str) -> list:
    """
    쿼리 문자열에서 최소 2글자 이상의 한글 단어를 추출하고,
    '추천해줘', '보여줘', '같은' 등 특정 불필요한 단어는 제거
    
    Args:
        query (str): 사용자 입력 쿼리

    Returns:
        list: 추출된 키워드 리스트
    """
    # 한글 정규식으로 최소 2글자 이상의 단어 추출
    keywords = re.findall(r'[가-힣]{2,}', query)
    # 필요하지 않은 단어들을 제외
    exclude = {"추천해줘", "보여줘", "같은"}
    return [w for w in keywords if w not in exclude]

def guess_category_from_keywords(keywords: list, categories: list) -> str:
    """
    추출된 키워드를 기반으로 후보 카테고리 목록에서 가장 잘 맞는 카테고리를 추정
    키워드 리스트를 역순으로 순회하여, 키워드가 카테고리 문자열 내에 포함되어 있으면 해당 카테고리를 반환
    없으면 기본값 "가구"를 반환
    
    Args:
        keywords (list): 쿼리에서 추출된 키워드 리스트
        categories (list): DB 등에서 조회된 카테고리 후보 리스트

    Returns:
        str: 추정된 카테고리
    """
    for kw in reversed(keywords):
        for cat in categories:
            if kw in cat:
                return cat
    return "가구"

def filter_products_by_category(products: list, category: str) -> list:
    """
    제품 목록에서 카테고리 문자열(소문자로 비교)을 기준으로 필터링
    
    Args:
        products (list): 제품 문서 리스트 (각 문서는 dict)
        category (str): 필터 기준이 되는 카테고리 문자열

    Returns:
        list: 해당 카테고리에 일치하는 제품 리스트
    """
    category = category.strip().lower()
    matched = []
    for p in products:
        cat = p.get("category", "").strip().lower()
        if category in cat:
            matched.append(p)
    # print(f"[DEBUG] 필터링된 카테고리: {category}, 실제 매칭된 개수: {len(matched)}", flush=True)
    return matched

def filter_by_query_keywords(products: list, query: str) -> list:
    """
    제품 목록에서 쿼리의 키워드가 제품의 name, description, detail 필드 중 어느 하나라도 포함되어 있는지 확인하여 필터링
    
    Args:
        products (list): 제품 문서 리스트 (각 문서는 dict)
        query (str): 사용자 입력 쿼리

    Returns:
        list: 키워드가 포함된 제품 리스트
    """
    keywords = extract_keywords_from_query(query)
    # print(f"[DEBUG] 추출된 키워드: {keywords}")
    filtered = []
    for p in products:
        searchable_text = f"{p.get('name', '')} {p.get('description', '')} {p.get('detail', '')}"
        if any(k in searchable_text for k in keywords):
            filtered.append(p)
    return filtered