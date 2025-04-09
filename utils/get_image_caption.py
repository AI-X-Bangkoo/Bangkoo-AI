import io
from PIL import Image
from google.generativeai import GenerativeModel
from model_loader import model_manager
import numpy as np

def get_image_caption_and_embedding(image: Image.Image):
    model = GenerativeModel("gemini-1.5-flash")

    # 이미지 byte 변환
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # 캡션 생성
    response = model.generate_content(
        [
            {"mime_type": "image/jpeg", "data": image_bytes},
            "이 이미지를 설명해줘. 어떤 물건인지, 형태, 색상, 용도를 한 문단으로 설명해줘. 결과는 검색 시스템에 사용될 예정이야."
        ]
    )
    caption = response.text.strip()
    print(f"[캡션 생성 완료] {caption}")

    # 텍스트 임베딩
    e5_model = model_manager.text_model  # SentenceTransformer 객체
    embedding = e5_model.encode([f"query: {caption}"], normalize_embeddings=True)

    return caption, embedding.astype(np.float32)
