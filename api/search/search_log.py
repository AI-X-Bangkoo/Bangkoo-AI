from datetime import datetime
from mongo_manager import mongo_manager

"""
최초 작성자: 김동규
최초 작성일: 2025-04-14

최근 검색어, 인기 검색어 함수 정의

"""


def save_search_log(query: str, user_id: str = "anonymous", source: str = "text"):
    if not query:
        return
    db = mongo_manager.db
    db["search_logs"].insert_one({
        "query": query.strip(),
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "source": source
    })



def get_recent_searches(user_id: str, limit=10):
    db = mongo_manager.db
    recent = db["search_logs"].find(
        {"user_id": user_id}, {"query": 1}
    ).sort("timestamp", -1).limit(limit)
    return list({doc["query"] for doc in recent})



def get_popular_searches(limit=10):
    db = mongo_manager.db
    pipeline = [
        {"$group": {"_id": "$query", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    popular = db["search_logs"].aggregate(pipeline)
    return [{"query": doc["_id"], "count": doc["count"]} for doc in popular]
