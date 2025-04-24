import os
import json
import time
from pymongo import MongoClient, UpdateOne
from datetime import datetime
from dotenv import load_dotenv
from model_loader import model_manager  #이미지 및 텍스트 임베딩 모델 로더
import numpy as np


"""
최초 작성자자 : 김병훈
최초작업일 : 2025-04-17

-관리자가 상품을 개별로 등록할때 
 몽고DB안에 누락돼는 임베딩값을 넣어주는 것을 
 자동화 할수 있는 파이프 라인 구축
"""


#.env 파일에서 환경 변수 로드
load_dotenv()

#MongoDB URI 로드 및 DB/컬렉션 설정
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["bangkoo"]
collection = db["products"]


#이미지 + 텍스트 임베딩 -> 정규화 -> 결합하는 함수까지지
def generate_combined_embedding(item):
    try:
        # 리스트로 저장된 임베딩을 numpy 배열로 변환
        image_emb = np.array(item["imageEmbedding"], dtype=np.float32)
        text_emb = np.array(item["textEmbedding"], dtype=np.float32)

        # 임베딩 차원이 기대값과 다르면 건너뜀
        if image_emb.shape[0] != 1024 or text_emb.shape[0] != 768:
            print(f"[경고] 임베딩 차원 오류: image({image_emb.shape[0]}), text({text_emb.shape[0]})")
            return None

        # 각 임베딩 정규화 (벡터 크기를 1로 맞춤)
        image_emb /= np.linalg.norm(image_emb)
        text_emb /= np.linalg.norm(text_emb)

        # 이미지:텍스트 = 0.7:0.3 비율로 가중 합쳐서 연결
        combined = np.concatenate([image_emb * 0.7, text_emb * 0.3])

        # 최종 벡터도 정규화
        combined /= np.linalg.norm(combined)

        # float32 배열로 변환 후 리스트로 리턴 (MongoDB 저장용)
        return combined.astype(np.float32).tolist()

    except Exception as e:
        print(f"[오류] combinedEmbedding 생성 실패: {e}")
        return None
    
    
#각 제품 item에 임베딩 필드를 생성하는 함수
def encode_embeddings(item):
    # 이미지 임베딩이 없고, imageUrl이 존재할 경우 이미지 임베딩 생성
    if "imageEmbedding" not in item and "imageUrl" in item:
        try:
            item["imageEmbedding"] = model_manager.encode_image_from_url(item["imageUrl"]).tolist()
        except Exception as e:
            print(f"[오류] 이미지 임베딩 실패: {e}")
            item["imageEmbedding"] = None  # 실패 시 None 처리
    
    # 텍스트 임베딩이 없을 경우 name + description + detail을 합쳐서 생성
    if "textEmbedding" not in item:
        try:
            query_text = f"{item.get('name', '')} {item.get('description', '')} {item.get('detail', '')}"
            item["textEmbedding"] = model_manager.text_model.encode(
                [f"query: {query_text}"], normalize_embeddings=True
            )[0].tolist()
        except Exception as e:
            print(f"[오류] 텍스트 임베딩 실패: {e}")
            item["textEmbedding"] = None  # 실패 시 None 처리

    # 두 임베딩이 다 있을 때만 combinedEmbedding 생성
    if "imageEmbedding" in item and "textEmbedding" in item:
        combined = generate_combined_embedding(item)
        if combined:
            item["combinedEmbedding"] = combined
        else:
            print(f"[경고] combinedEmbedding 생성 실패: {item['link']}")
            item["combinedEmbedding"] = None  # combinedEmbedding 생성 실패 시 None 처리

    return item



# 실제 업로드 처리 함수(임베딩 포함 MongoDB upsert)
def upload_products(products: list):
    ops = []

    for item in products:
        # 'link' 필드가 없으면 에러 발생
        if "link" not in item:
            print(f"[오류] link 필드 누락: {item}")
            continue  # link 필드가 없으면 해당 항목을 건너뜁니다.

        item = encode_embeddings(item)  # 임베딩 생성
        item["updateAt"] = datetime.utcnow()  # 수정 시간 갱신

        # link 필드를 기준으로 upsert(존재하면 업데이트, 없으면 삽입)
        ops.append(UpdateOne(
            {"link": item["link"]},
            {"$set": item, "$setOnInsert": {"createAt": datetime.utcnow()}},
            upsert=True
        ))

    # 일괄 반영(bulk_write)
    if ops:
        result = collection.bulk_write(ops)
        print(f"[업로드 결과] 삽입: {result.upserted_count} / 수정: {result.modified_count}")
    else:
        print("[알림] 업로드 할 항목이 없습니다.")

    ops = []

    for item in products:
        item = encode_embeddings(item)  #임베딩 생성
        item["updateAt"] = datetime.utcnow() #수정 시간 갱신

        #link 필드를 기준으로 upsert(존재하면 업데이트, 없으면 삽임)
        ops.append(UpdateOne(
            {"link": item["link"]},
            {"$set": item, "$setOnInsert" : {"createAt" : datetime.utcnow()}},
            upsert=True
        ))

    #일괄 반영(bulk_write)
    if ops:
        result = collection.bulk_write(ops)
        print(f"[업로드 결과] 삽입: {result.upserted_count} / 수정 : {result.modified_count}")
    else:
        print("[알림] 업로드 할 항목이 없습니다.")


#entry point - json 파일 읽고 업로드 시작
if __name__ == "__main__":
    with open("new_products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    upload_products(products)