from pymongo import MongoClient
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Mongo 연결
client = MongoClient(MONGO_URI)
db = client["bangkoo"]

BASE_DIR = Path(__file__).resolve().parent

# ============================
# 1. 제품 데이터 업로드
# ============================

collection = db["products"]

# with open("json/clip_metadata_v3.json", encoding="utf-8") as f:
#     metadata = json.load(f)

with open(BASE_DIR / "json" / "clip_metadata_v3.json", encoding="utf-8") as f:
    metadata = json.load(f)
    
# with open("json/ko_synonyms.json", encoding="utf-8") as f:
#     synonyms = json.load(f)
    
with open(BASE_DIR / "json" / "ko_synonyms.json", encoding="utf-8") as f:
    metadata = json.load(f)

image_embeddings = np.load(BASE_DIR / "npy" / "image_embeddings.npy")
text_embeddings = np.load(BASE_DIR / "npy" / "product_text_embeddings_v3.npy")
# text_embeddings = np.load("npy/product_text_embeddings_v3.npy")

for i, item in enumerate(metadata):
    doc = {
        "name": item["이름"],
        "description": item["설명"],
        "detail": item.get("상세설명"),
        "price": item.get("할인가") or item.get("정상가"),
        "link": item["링크"],
        "imageUrl": item["이미지"],
        "csv": item["csv"],
        "imageEmbedding": image_embeddings[i].astype(float).tolist(),
        "textEmbedding": text_embeddings[i].astype(float).tolist(),
        "createdAt": datetime.utcnow()
    }
    collection.insert_one(doc)

print(f"제품 {len(metadata)}개 업로드 완료")

# ============================
# 2. 동의어 사전 업로드
# ============================

# 기존 데이터 삭제 후 새로 저장 (선택)
db["synonyms"].delete_many({})
db["synonyms"].insert_one({"_id": "korean", "dict": synonyms})

print("동의어 사전 업로드 완료")
