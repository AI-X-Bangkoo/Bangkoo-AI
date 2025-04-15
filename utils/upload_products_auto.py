import os
import json
import time
from pymongo import MongoClient, UpdateOne
from datetime import datetime
from dotenv import load_dotenv
from model_loader import model_manager
import numpy as np

<<<<<<< HEAD
=======
"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

- 제품 정보를 MongoDB에 업로드하면서 누락된 임베딩을 자동 생성
- imageEmbedding: 이미지 URL을 통해 추론
- textEmbedding: name + description + detail을 기반으로 생성
- createdAt / updatedAt 필드를 기준으로 upsert
- 중복 여부는 link 필드를 기준으로 판단
- bulk_write로 MongoDB에 일괄 반영
"""


>>>>>>> eaa1fc8391c3bb9030bc37fb618076e66a28c39f
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["bangkoo"]
collection = db["products"]

def encode_embeddings(item):
    # 임베딩 없을 경우 생성
    if "imageEmbedding" not in item and "imageUrl" in item:
        item["imageEmbedding"] = model_manager.encode_image_from_url(item["imageUrl"]).tolist()
    if "textEmbedding" not in item:
        query_text = f"{item.get('name', '')} {item.get('description', '')} {item.get('detail', '')}"
        item["textEmbedding"] = model_manager.text_model.encode([f"query: {query_text}"], normalize_embeddings=True)[0].tolist()
    return item

def upload_products(products: list):
    ops = []
    for item in products:
        item = encode_embeddings(item)
        item["updatedAt"] = datetime.utcnow()
        ops.append(UpdateOne(
            {"link": item["link"]},  # 링크 기준 중복 판단
            {"$set": item, "$setOnInsert": {"createdAt": datetime.utcnow()}},
            upsert=True
        ))

    if ops:
        result = collection.bulk_write(ops)
        print(f"[업로드 결과] 삽입: {result.upserted_count} / 수정: {result.modified_count}")
    else:
        print("[알림] 업로드할 항목이 없습니다.")

if __name__ == "__main__":
    # 새로운 제품은 양식에 맞춰 new_products.json 파일로 만들어서 같은 경로에 두면 됨
    with open("new_products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    upload_products(products)
