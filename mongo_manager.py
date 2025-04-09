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
            socketTimeoutMS=100000,
            connectTimeoutMS=100000,
            serverSelectionTimeoutMS=100000,
        )
        self.db = self.client["bangkoo"]
        self.products = self.db["products"]
        self.ready = True
        print("[MongoDB] 연결 완료")

mongo_manager = MongoDBManager()
