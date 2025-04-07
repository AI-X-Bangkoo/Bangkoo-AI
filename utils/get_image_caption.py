import base64
from google.generativeai import GenerativeModel

# 이미지 캡셔닝 함수
def get_image_caption(image: Image.Image) -> str:
    model = GenerativeModel("gemini-1.5-flash")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    response = model.generate_content(
        [
            {"mime_type": "image/jpeg", "data": image_bytes},
            "이 이미지를 설명해줘. 어떤 물건인지, 형태, 색상, 용도를 한 문단으로 설명해줘. 결과는 검색 시스템에 사용될 예정이야."
        ]
    )
    return response.text.strip()
