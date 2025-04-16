import re

"""
최초 작성자: 김동규
최초 작성일: 2025-04-05

- Gemini 응답으로부터 JSON 형태의 추천 결과만 추출
- ```json ... ``` 형식 또는 순수 JSON 배열 형태만 파싱 대상
- fallback 처리로 파싱 실패 시 전체 텍스트 반환
- JSON 구조가 깨졌을 경우 대응을 위해 사용
"""

def extract_json_from_markdown(text: str) -> str:
    """
    Gemini 응답에서 ```json ... ``` 또는 순수 JSON 배열만 추출합니다.

    Args:
        text (str): Gemini 응답 문자열

    Returns:
        str: JSON만 추출된 문자열
    """
    # 1. ```json ``` 블록 추출
    json_blocks = re.findall(r"```json(.*?)```", text, re.DOTALL)
    if json_blocks:
        return json_blocks[0].strip()

    # 2. 순수 JSON 배열 형태 추출
    array_blocks = re.findall(r"(\[\s*{.*?}\s*\])", text, re.DOTALL)
    if array_blocks:
        return array_blocks[0].strip()

    # 3. fallback → 전체 반환 (디버깅용)
    return text.strip()
