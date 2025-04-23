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
        print(f"[DEBUG] Gemini 판단 결과: {answer}")
        return "예" in answer or "yes" in answer
    except Exception as e:
        print("[ERROR] Gemini 판단 실패:", e)
        return False


"""
함수 추가자 : 김병훈
작성일 : 2025-04-23

-gemini한테 방 이미지 스타일 분석을 위한 
 프로프트 요청 함수
"""
# 기존 함수 수정: 이미지 파일을 받아서 분석
async def analyze_room_with_gemini_by_file(file: UploadFile) -> dict:
    """
    업로드된 이미지 파일을 사용하여 Gemini 스타일 분석
    :param file: 업로드된 이미지 파일
    :return: 분석된 스타일 정보
    """
    prompt = build_style_prompt()  # 스타일 분석을 위한 기본 프롬프트

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        

        # 이미지를 Gemini에 맞는 포맷으로 변환 (JPEG 등)
        with BytesIO() as img_io:
            image.save(img_io, format="JPEG")
            img_io.seek(0)
            image_data = img_io.getvalue()
       

            # Gemini 모델에 비동기로 이미지와 함께 요청
        
            response = await gemini_model.generate_content_async(
                [prompt, {"file": image_data, "mime_type": "image/jpeg"}],
                generation_config={"temperature": 0.4}
            )
       

        text = response.text.strip()
 
                
        # JSON 파싱 (스타일 관련 정보 추출)
        parsed = extract_and_parse_json(text)
    
        return parsed[0] if isinstance(parsed, list) else parsed

    except Exception as e:
        return {
            "style": "unknown",
            "color_palette": [],
            "furniture_types": [],
            "materials": [],
            "lighting_mood": "",
            "layout_features": "",
            "decor_items": []
        }