import json
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

# 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 모델 로드
clip_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device).eval().to(torch.float32)
clip_processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
text_model = SentenceTransformer("intfloat/e5-base-v2", device=device)

# client = MongoClient("mongodb://localhost:27017")
# db = client["bangkoo"]
# products = list(db["products"].find())
# 동의어 불러오기
# synonyms = db["synonyms"].find_one({"_id": "korean"})["dict"]

# # 임베딩 추출
# image_embeddings = np.array([p["imageEmbedding"] for p in products])
# text_embeddings = np.array([p["textEmbedding"] for p in products])
# items = products  # 기존의 metadata 역할

# 데이터 로드
with open("clip_metadata_v3.json", encoding="utf-8") as f:
    items = json.load(f)

with open("ko_synonyms.json", encoding="utf-8") as f:
    synonyms = json.load(f)

# 임베딩 불러오기
image_embeddings = np.load("image_embeddings.npy")  # shape: (N, D)
text_embeddings = np.load("product_text_embeddings_v3.npy")  # shape: (N, D)

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
            "이름": item["이름"],
            "설명": item["설명"],
            "링크": item["링크"],
            "이미지": item["이미지"]
        })

    return results
