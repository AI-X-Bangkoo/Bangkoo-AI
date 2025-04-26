import os
import time
import pymongo
import numpy as np
import schedule
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["bangkoo"]

def update_reranking_weights():
    """
    후보 피드백 데이터를 집계하여 재정렬에 사용할 가중치(벡터, BM25, LLM bonus, 피드백)를 업데이트
    집계 결과에 따라 다른 A/B Variant를 적용할 수도 있음
    """
    feedback_col = db["candidate_feedback"]
    config_col = db["search_config"]

    # 모든 후보의 평균 CTR을 계산 (CTR = clicks / (impressions + 1))
    pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_ctr": {"$avg": {"$divide": ["$clicks", {"$add": ["$impressions", 1]}]}}
            }
        }
    ]
    result = list(feedback_col.aggregate(pipeline))
    
    if result:
        avg_ctr = result[0]["avg_ctr"]
        # print(f"[Update Weights] 현재 평균 CTR: {avg_ctr:.3f}")
        
        # A/B Variant 선택: CTR 값에 따라 다른 가중치 세트를 적용하는 예시임
        if avg_ctr > 0.1:
            new_weights = {"vector": 0.45, "bm25": 0.25, "llm": 0.15, "feedback": 0.15}
        else:
            new_weights = {"vector": 0.5, "bm25": 0.3, "llm": 0.2, "feedback": 0.0}
        
        # 업데이트된 가중치를 MongoDB의 search_config 컬렉션에 저장 (config 키 reranking_weights 사용)
        config_col.update_one(
            {"config": "reranking_weights"},
            {"$set": new_weights},
            upsert=True
        )
        # print(f"[Update Weights] 재정렬 가중치 업데이트됨: {new_weights}")
    else:
        print("[Update Weights] 피드백 데이터가 부족합니다.")

# 스케줄러를 사용하여 매일 정해진 시각(예: 02:00)에 업데이트 작업을 실행
schedule.every().day.at("02:00").do(update_reranking_weights)

if __name__ == "__main__":
    # print("[Feedback Pipeline] 피드백 가중치 업데이트 파이프라인 시작")
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 스케줄 확인
