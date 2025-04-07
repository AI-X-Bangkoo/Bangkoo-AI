import json
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import os

"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

하이브리드 검색 모듈

- E5 기반 텍스트 임베딩 + CLIP 기반 이미지 임베딩 결합
- 동의어 확장 → 임베딩 계산 → 유사도 기반 검색 수행
- MongoDB에서 제품 정보 및 임베딩 불러옴
"""


# 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 모델 로드
clip_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device).eval().to(torch.float32)
clip_processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
text_model = SentenceTransformer("intfloat/e5-base-v2", device=device)

# MongoDB 연결
MONGO_URI = os.getenv("MONGO_URI")

# print(f"현재 MONGO_URI: {MONGO_URI}")

client = MongoClient(MONGO_URI)
db = client["bangkoo"]
product_collection = db["products"]
# synonyms = db["synonyms"].find_one({"_id": "korean"})["dict"]

# 동의어 사전 로딩
doc = db["synonyms"].find_one({"_id": "korean"})
# print("MongoDB로부터 받아온 문서:", doc)

if doc is None:
    raise ValueError("'_id': 'korean' 문서가 존재하지 않음")
if "dict" not in doc:
    raise ValueError("'dict' 필드가 존재하지 않음")

synonyms = doc["dict"]
# print("동의어 사전 로딩 완료. 총 항목 수:", len(synonyms))



# 전체 데이터 로딩
products = list(product_collection.find())
items = products
image_embeddings = np.array([p["imageEmbedding"] for p in products], dtype=np.float32)
text_embeddings = np.array([p["textEmbedding"] for p in products], dtype=np.float32)

# 동의어 확장 함수
def expand_query(query, synonyms):
    words = query.split()
    expanded = []
    for word in words:
        if word in synonyms:
            expanded.append([word] + synonyms[word])
        else:
            expanded.append([word])
    from itertools import product
    return list(set([' '.join(combo) for combo in product(*expanded)] + [query]))

# 임베딩 함수
def get_text_embedding(text):
    return text_model.encode([f"query: {text}"], normalize_embeddings=True)

def get_clip_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


# 동의어 확장 함수
def expand_query(query, synonyms):
    words = query.split()
    expanded = []
    for word in words:
        if word in synonyms:
            expanded.append([word] + synonyms[word])
        else:
            expanded.append([word])
    # 모든 조합 생성
    from itertools import product   
    candidates = [' '.join(combo) for combo in product(*expanded)]
    return list(set([query] + candidates))

# 임베딩 함수
def get_text_embedding(text):
    return text_model.encode([f"query: {text}"], normalize_embeddings=True)

def get_clip_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

# 검색 함수
def hybrid_search(query, top_k=None):
    queries = expand_query(query, synonyms)
    print(f"동의어 확장 결과: {queries}")

    # 가장 높은 점수의 조합을 사용
    best_score = -1
    best_indices = []
    best_sim = None  # 전체 유사도 점수 저장용

    for q in queries:
        e5_embed = get_text_embedding(q)
        clip_embed = get_clip_text_embedding(q)

        sim_text = cosine_similarity(e5_embed, text_embeddings)[0]
        sim_image = cosine_similarity(clip_embed, image_embeddings)[0]
        sim = 0.6 * sim_text + 0.4 * sim_image

        # top_idx = np.argsort(sim)[::-1][:top_k] # top_k로 자르기, 검색 함수 인자 수정 필요
        top_idx = np.argsort(sim)[::-1]  # 유사도 순 전체 정렬
        if sim[top_idx[0]] > best_score:
            best_score = sim[top_idx[0]]
            best_indices = top_idx
            best_sim = sim

    results = []
    for i in best_indices:
        item = items[i]
        results.append({
            "이름": item["name"],
            "설명": item["description"],
            "상세설명": item.get("detail", ""),
            "링크": item["link"],
            "이미지": item["imageUrl"],
            "할인가": item.get("price", "정보 없음"),
            "정상가": item.get("price", "정보 없음"),
            "csv": item.get("csv", "")
        })


    return results
