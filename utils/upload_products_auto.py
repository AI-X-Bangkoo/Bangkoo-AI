import os
import json
import time
from pymongo import MongoClient, UpdateOne
from datetime import datetime
from dotenv import load_dotenv
from model_loader import model_manager
import numpy as np

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
