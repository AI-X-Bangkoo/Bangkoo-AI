from pymongo import MongoClient
from dotenv import load_dotenv
import os

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.products = None
        self.ready = False

    def connect(self):
        load_dotenv()
        MONGO_URI = os.getenv("MONGO_URI")
        self.client = MongoClient(
            MONGO_URI,
            socketTimeoutMS=300000,
            connectTimeoutMS=300000,
            serverSelectionTimeoutMS=300000,
        )
        self.db = self.client["bangkoo"]
        self.products = self.db["products"]
        self.ready = True
        print("[MongoDB] 연결 완료")
        
    def get_products(self, filter_query=None, projection=None):
        # 기본 projection 설정 (필요한 필드만 가져오기)
        if projection is None:
            projection = {
                "name": 1, "description": 1, "detail": 1,
                "imageEmbedding": 1, "textEmbedding": 1,
                "link": 1, "imageUrl": 1, "price": 1,
                "category": 1, "csv": 1, "model3dUrl": 1
            }
        if filter_query is None:
            filter_query = {}
        return list(self.products.find(filter_query, projection))

mongo_manager = MongoDBManager()
