import json
import re


"""
최초 작성자 : 김병훈
최초 작성일 : 2025-04-23

응답이 항상 완벽한 json형식으로 안올 것을 대비한 
전처리 작업 
"""
def extract_and_parse_json(text:str)->dict:
    """
    GEMINI 응답에서 JSON 부분만 추찰하여 안전하게 파싱
    """

    try:
        #중괄호 배열 형태 JSON 을 찾아냄 : [ {...},{...}]
        json_match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
        if not json_match:
            print("[waring] JSON 형태를 찾지 못하였습니다. 원봉 응답:\n", text)
            return get_default_analysis()
        
        json_str = json_match.group(0)

        parsed = json.loads(json_str)
        return parsed[0] if isinstance(parsed, list) else parsed
    
    except Exception as e :
        print("[ERROR]  JSON 추출 또는 파싱 실패:", e)
        return get_default_analysis()
    
def get_default_analysis() -> dict:
    """
    파싱 실패 시 기본 반환값
    """
    return {
        "style": "unknown",
        "color_palette": [],
        "furniture_types": [],
        "materials": [],
        "lighting_mood": "",
        "layout_features": "",
        "decor_items": []
    }