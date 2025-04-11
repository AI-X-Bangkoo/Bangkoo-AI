import re
from urllib.parse import unquote, urlparse, parse_qs

"""
최초 작성자: 김동규
최초 작성일: 2025-04-07

- 네이버, 구글 이미지 검색 결과에서 원본 이미지 주소를 추출
- imgurl, src 파라미터를 기준으로 실제 이미지(.jpg, .png 등) 링크를 반환
- fallback: 원본 URL 그대로 반환
"""


def extract_direct_image_url(possible_url: str) -> str:
    """
    네이버, 구글 이미지 주소에서 실제 이미지 링크 (.jpg, .png 등) 추출
    """
    # 바로 이미지 확장자 있으면 그대로 반환
    if re.search(r"\.(jpg|jpeg|png|gif|webp)(\?|$)", possible_url, re.IGNORECASE):
        return possible_url

    parsed = urlparse(possible_url)
    query_params = parse_qs(parsed.query)

    # 네이버: src=에 진짜 이미지 URL
    if "src" in query_params:
        actual_url = unquote(query_params["src"][0])
        if re.search(r"\.(jpg|jpeg|png|gif|webp)(\?|$)", actual_url, re.IGNORECASE):
            return actual_url

    # 구글: imgurl=에 진짜 이미지 URL
    if "imgurl" in query_params:
        actual_url = unquote(query_params["imgurl"][0])
        if re.search(r"\.(jpg|jpeg|png|gif|webp)(\?|$)", actual_url, re.IGNORECASE):
            return actual_url

    # fallback: 원본 반환
    return possible_url
