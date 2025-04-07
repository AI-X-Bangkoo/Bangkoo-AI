from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["bangkoo"]
product_collection = db["products"]

print("전체 개수:", product_collection.count_documents({}))
print("샘플 하나:", product_collection.find_one())
