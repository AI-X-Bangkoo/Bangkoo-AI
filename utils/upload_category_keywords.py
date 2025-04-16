import os
from pymongo import MongoClient
from dotenv import load_dotenv

"""
최초 작성자: 김동규
최초 작성일: 2025-04-04

- category_keywords 사전을 MongoDB에 저장하는 초기화 스크립트
- 'dict' 필드를 포함한 JSON 구조로 upsert 수행
- 기존 값이 있으면 갱신, 없으면 새로 삽입
- 검색 필터링이나 Gemini 분류 기준으로 활용
"""

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["bangkoo"]

CATEGORY_KEYWORDS = {
    "의자": ["의자", "체어", "스툴", "암체어", "암의자"],
    "가전제품": ["인덕션", "식기세척기", "레인지", "오븐"],
    "테이블": ["식탁", "테이블", "다이닝", "바테이블"],
    "침대/매트리스": ["침대", "매트리스", "프레임", "베드", "헤드보드"],
    "이불/베개": ["이불", "베개"],
    "소파": ["소파", "카우치", "리클라이너"],
    "쿠션/담요": ["쿠션", "담요"],
    "수납": ["수납", "서랍", "장", "선반", "책장", "캐비닛", "옷장", "콘솔", "유닛", "박스", "바구니", "가방"],
    "아웃도어": ["야외", "아웃도어", "정원", "실외", "피크닉", "벤치"],
    "화분": ["화분", "플랜터", "화병", "꽃병", "식물"],
    "데코": ["액자", "장식", "데코", "오브제", "캔들", "조화"],
    "러그": ["러그", "카페트", "매트"],
    "조명": ["조명", "램프", "등", "스탠드", "샹들리에"]
}

# 덮어쓰기용 upsert
db["category_keywords"].update_one(
    {"_id": "korean"},
    {"$set": {"dict": CATEGORY_KEYWORDS}},
    upsert=True
)

print("카테고리 키워드 업로드 완료")
