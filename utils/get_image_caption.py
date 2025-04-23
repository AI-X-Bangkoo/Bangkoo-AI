import io
from PIL import Image
from google.generativeai import GenerativeModel
from model_loader import model_manager
from mongo_manager import mongo_manager
import numpy as np
import torch

"""
최초 작성자: 김동규
최초 작성일: 2025-04-09

- 이미지 한 장을 Gemini에 입력하여 설명과 카테고리를 추출
- 응답에서 '설명'과 '카테고리'를 파싱하여 텍스트 임베딩 생성
- category_keywords에 정의된 카테고리 목록 중 하나만 선택되도록 유도
- 임베딩은 e5-base-v2 모델을 사용하여 정규화 후 float32로 반환
- 이미지 임베딩은 CLIP 기반 jina-clip-v2 모델을 사용하여 추출
"""


def get_image_caption_and_embedding(image: Image.Image):
    model = GenerativeModel("models/gemini-2.0-flash")

    # 1. 이미지 byte 변환
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # 2. category_keywords 문서에서 키만 추출
    if not mongo_manager.ready:
        mongo_manager.connect()
    db = mongo_manager.db
    category_doc = db["category_keywords"].find_one()
    if not category_doc or "dict" not in category_doc:
        raise ValueError("category_keywords 문서가 존재하지 않거나 잘못된 구조입니다.")

    valid_categories = list(category_doc["dict"].keys())
    category_list_str = ", ".join(valid_categories)

    # 3. 프롬프트 구성
    prompt = (
        f"다음 이미지를 설명해줘. 반드시 어떤 물건인지, 전체적인 형태, 색상, 용도, 크기를 포함한 설명을 순서대로 한 문단으로 해줘.\n"
        f"그리고 반드시 아래의 카테고리 목록 중에서 이 물건에 가장 적절한 하나의 카테고리만 골라줘. "
        f"절대 목록에 없는 단어를 말하지 말고, 반드시 이 목록 중 하나를 그대로 골라야 해.\n\n"
        f"{category_list_str}\n\n"
        f"출력 형식은 반드시 아래처럼 해:\n"
        f"설명: (한 문단)\n"
        f"카테고리: (위 목록 중 정확히 하나)"
    )

    # 4. Gemini 응답 받기
    response = model.generate_content([
        {"mime_type": "image/jpeg", "data": image_bytes},
        prompt
    ])
    text = response.text.strip()
    print(f"[캡션+카테고리 생성 완료]\n{text}")

    # 5. 설명, 카테고리 파싱
    description, category = "", ""
    for line in text.splitlines():
        if line.strip().lower().startswith("설명:"):
            description = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("카테고리:"):
            category = line.split(":", 1)[1].strip()

    # 6. 텍스트 임베딩 생성
    e5_model = model_manager.text_model
    embedding = e5_model.encode([f"query: {description}"], normalize_embeddings=True)

    return description, embedding.astype(np.float32), category

def get_image_embedding(image: Image.Image):
    if not model_manager.ready:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")

    clip_model = model_manager.clip_model
    clip_processor = model_manager.clip_processor
    device = model_manager.device

    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()
