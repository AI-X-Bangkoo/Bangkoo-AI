from pymongo import MongoClient
from dotenv import load_dotenv
import os

"""
최초 작성자: 김동규
최초 작성일: 2025-04-07
설명: MongoDBManager 클래스를 정의하여 모델 및 DB 초기화를 lazy-load 또는 FastAPI startup 이벤트에서 처리할 수 있도록 구성
"""

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
        
mongo_manager = MongoDBManager()
