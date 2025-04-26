import os
import google.generativeai as genai
from dotenv import load_dotenv
from utils import build_style_prompt
from utils import extract_and_parse_json
from fastapi import UploadFile
from io import BytesIO
from PIL import Image


"""
최초 작성자: 김동규
최초 작성일: 2025-04-09


- 사용자의 쿼리가 이미지에 의존하는지 자동 판단

"""

load_dotenv()


# 초기화
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

def should_use_image_for_recommendation(query: str) -> bool:
    """
    Gemini를 사용하여 사용자 쿼리가 이미지 기반인지 판단
    """
    prompt = f"""
사용자의 쿼리: "{query}"

이 쿼리는 업로드된 방 사진이나 이미지 속 공간을 참조하고 있습니까?

- 예: "여기에 어울리는 가구 추천", "이 방에 맞는 의자", "이런 분위기와 잘 어울리는"
- 아니오: "모던한 소파 추천", "화이트 톤 식탁", "미니멀한 책장"

"예" 또는 "아니오"로만 대답하세요.
"""

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip().lower()
        # print(f"[DEBUG] Gemini 판단 결과: {answer}")
        return "예" in answer or "yes" in answer
    except Exception as e:
        # print("[ERROR] Gemini 판단 실패:", e)
        return False
