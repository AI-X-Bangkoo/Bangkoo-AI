# api/autoRecommend/vector_index.py
import faiss
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os

#작성자: 김병훈
#DB에서 전체 가구 벡터를 매번 순회하지 않고 초고속으로 tpo-k 결과를 가져오기 위한 작업

class VectorIndex:
    def __init__(self):
        self.index = None
        self.product_docs = []  # 인덱스 순서대로 doc 리스트

    def build(self):
         # 1) MongoDB에서 모든 textEmbedding 불러오기
        load_dotenv()
        client = MongoClient(os.getenv("MONGO_URI"), 
                            socketTimeoutMS=300000,
                            connectTimeoutMS=300000,
                            serverSelectionTimeoutMS=300000)
        db = client["bangkoo"]
        cursor = db["products"].find(
            {"textEmbedding": {"$exists": True}},
            {"textEmbedding": 1, "name": 1, "price": 1, "category": 1,
            "link": 1, "imageUrl": 1, "description": 1}
        )

        embeddings = []
        valid_docs = []

        for doc in cursor:
            emb = doc.get("textEmbedding")
            # 필터링
            if emb is None:
                continue
            if not isinstance(emb, list):
                continue
            if len(emb) != 768:
                continue
            # 모두 통과하면 추가
            embeddings.append(emb)
            valid_docs.append(doc)

        if not embeddings:
            raise ValueError("❗ 유효한 임베딩이 없습니다. build() 실패")

        embeddings = np.array(embeddings, dtype="float32")
        self.product_docs = valid_docs

        # 2) (Inner Product 기준) 벡터 정규화
        faiss.normalize_L2(embeddings)

        # 3) 인덱스 생성 → Flat(IP)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)


    def query(self, query_vec: np.ndarray, top_k: int = 10):
        # query_vec: (1, dim) float32, 이미 normalize_L2 됐다 가정
        D, I = self.index.search(query_vec, top_k)
        return [(self.product_docs[i], float(D[0][j])) 
                for j, i in enumerate(I[0])]

# 글로벌 객체로 한 번만 빌드
vector_index = VectorIndex()
vector_index.build()
print("✅ Faiss 인덱스 빌드 완료, 총 문서:", len(vector_index.product_docs))
