import os
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from io import BytesIO
from PIL import Image
from google import genai
# import google.generativeai as genai
from google.genai import types
# from google.generativeai import types
from utils.build_style_prompt import build_style_prompt
from utils.extract_and_parse_json import extract_and_parse_json

"""
최초 작성자 : 김병훈
최초 작성일 : 2025-04-24


- GenAI Python SDK 최신 async 호출 방식으로 수정
- AsyncModels.generate_content의 `temperature` 제거, 대신 `config`로 설정
"""

# 1. 환경 변수 로드 및 GenAI 클라이언트 초기화
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"), vertexai=False)
# 2. 비동기 모델 핸들러 분리
models_async = client.aio.models

async def analyze_room_with_gemini_by_file(file: UploadFile) -> dict:
    """
    업로드된 이미지 파일을 Gemini 모델에 보내 방 스타일 분석을 수행합니다.
    :param file: FastAPI UploadFile 객체
    :return: 분석 결과 딕셔너리
    """
    allowed_mime_types = ["image/jpg", "image/jpeg", "image/png"]

    # 1) 파일 검증
    if not file or file.content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 타입입니다: {file.content_type}"
        )

    try:
        # 2) 이미지 바이트 읽기 및 PIL 로딩
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 3) JPEG로 재인코딩
        with BytesIO() as img_io:
            image.save(img_io, format="JPEG")
            img_io.seek(0)
            jpeg_data = img_io.read()

        # 4) GenAI 멀티모달 파트 생성
        image_part = types.Part.from_bytes(
            data=jpeg_data,
            mime_type="image/jpeg"
        )

        # 5) 스타일 분석 프롬프트 생성
        prompt = build_style_prompt()
        print("********",prompt)

        # 6) Gemini 모델 비동기 호출
        # temperature 키워드 대신 GenerateContentConfig 사용
        config = types.GenerateContentConfig(temperature=0.4)
        response = await models_async.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part],
            config=config
        )

        # 7) 응답 텍스트 파싱
        text = response.text.strip()
        print("********",text)
        parsed = extract_and_parse_json(text)
        print("******",parsed)
        return parsed[0] if isinstance(parsed, list) else parsed

    except HTTPException:
        # 이미 HTTPException으로 래핑된 예외는 재전달
        raise
    except Exception as e:
        # 기타 예외는 상세 로그 후 HTTPException으로 변환
        print(f"[ERROR] Gemini 이미지 분석 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini 이미지 분석 실패: {e}"
        )
