import numpy as np
from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.image_analysis_utils import analyze_room_with_gemini_by_file
from api.autoRecommend.vector_index import vector_index
from fastapi import UploadFile, HTTPException
from typing import List, Optional
import time

async def recommend_furniture_for_room(
    file: UploadFile,
    style_keywords: Optional[List[str]] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    top_k: int = 100  # 추천 상위 개수 기본 100
):
    """
    방 이미지와 스타일 키워드, 가격 범위에 따라 최상위 top_k 가구를 추천합니다.
    :param file: UploadFile image
    :param style_keywords: 추가 스타일 키워드
    :param min_price: 최소 가격
    :param max_price: 최대 가격
    :param top_k: 추천할 상위 결과 개수
    :return: 추천된 가구 리스트
    """
    try:
        # 1) 방 이미지 분석
        t0 = time.time()
        print("1) 방 이미지 분석 시작")
        analysis = await analyze_room_with_gemini_by_file(file)
        elapsed_analysis = time.time() - t0
        print(f"1) 분석 완료: {elapsed_analysis:.2f}s, 결과: {analysis}")

        # 분석 결과 추출
        room_style    = analysis.get("style", "unknown")
        palette       = analysis.get("color_palette", [])
        furn_types    = analysis.get("furniture_types", [])
        materials     = analysis.get("materials", [])
        mood          = analysis.get("lighting_mood", "")
        layout        = analysis.get("layout_features", "")
        decor         = analysis.get("decor_items", [])

        # 2) 스타일 설명 및 임베딩
        t1 = time.time()
        desc_parts = [
            f"style: {room_style}",
            f"colors: {', '.join(palette)}",
            f"furniture: {', '.join(furn_types)}",
            f"materials: {', '.join(materials)}",
            f"mood: {mood}",
            f"layout: {layout}",
            f"decor: {', '.join(decor)}"
        ]
        if style_keywords:
            desc_parts.append(f"keywords: {', '.join(style_keywords)}")
        style_desc = " | ".join(desc_parts)
        print("2) 스타일 설명 텍스트:", style_desc)

        embedding = model_manager.text_model.encode(
            [style_desc], normalize_embeddings=True
        )  # already L2-normalized
        elapsed_embed = time.time() - t1
        print(f"2) 임베딩 생성 완료: {elapsed_embed:.2f}s, shape: {embedding.shape}")

        # 3) MongoDB 로드 및 필터링
        t2 = time.time()
        if not mongo_manager.ready:
            mongo_manager.connect()
        cursor = mongo_manager.db["products"].find(
            {"textEmbedding": {"$exists": True}},
            {"name":1, "price":1, "category":1,
             "textEmbedding":1, "link":1,
             "imageUrl":1, "description":1}
        )

        products, vectors = [], []
        for doc in cursor:
            try:
                price = int(doc.get("price", "0").replace(",", "").strip())
            except ValueError:
                price = 0
            if (min_price is not None and price < min_price) or \
               (max_price is not None and price > max_price):
                continue
            products.append(doc)
            vectors.append(doc["textEmbedding"])
        elapsed_load = time.time() - t2
        print(f"3) 상품 로드 및 필터링 완료: {elapsed_load:.2f}s, 개수: {len(products)}")
        if not products:
            return [{"이름":"추천 실패", "추천이유":"조건에 맞는 제품이 없습니다."}]

        # 4) 유사도 검색 (Vector Index)
        t3 = time.time()
        # embedding을 float32로 변환하여 인덱스 조회
        hits = vector_index.query(embedding.astype("float32"), top_k=top_k)
        elapsed_search = time.time() - t3
        print(f"4) 벡터 검색 완료: {elapsed_search:.2f}s, 상위 {len(hits)}개")

        # 5) 결과 포맷
        t4 = time.time()
        results = []
        for doc, score in hits:
            results.append({
                "이름":     doc.get("name", ""),
                "설명":     doc.get("description", ""),
                "링크":     doc.get("link", ""),
                "이미지":   doc.get("imageUrl", ""),
                "가격":     doc.get("price", ""),
                "추천이유": f"{room_style} 스타일과 유사도 {score:.3f}"
            })
        elapsed_format = time.time() - t4
        print(f"5) 결과 포맷 완료: {elapsed_format:.2f}s, 개수: {len(results)}")

        total_time = time.time() - t0
        print(f"전체 파이프라인 소요: {total_time:.2f}s")
        return results

    except Exception as e:
        print("❌ 가구 추천 오류:", e)
        raise HTTPException(status_code=500, detail=f"추천 오류: {e}")
