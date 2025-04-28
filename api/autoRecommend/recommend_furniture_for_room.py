import numpy as np
from fastapi import UploadFile, HTTPException
from typing import List, Optional
import time

from model_loader import model_manager
from mongo_manager import mongo_manager
from utils.image_analysis_utils import analyze_room_with_gemini_by_file
from api.autoRecommend.vector_index import vector_index

async def recommend_furniture_for_room(
    file: UploadFile,
    style_keywords: Optional[List[str]] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    top_k: int = 100  # 추천 상위 개수 기본 100
):
    """
    방 이미지와 스타일 키워드, 가격 범위에 따라 최상위 top_k 가구를 추천합니다.
    """
    try:
        # 1) 방 이미지 분석
        t0 = time.time()
        print("1) 방 이미지 분석 시작")
        analysis = await analyze_room_with_gemini_by_file(file)
        print(f"1) 분석 완료: {time.time() - t0:.2f}s, 결과: {analysis}")

        # 추출된 분석 결과
        room_style = analysis.get("style", "unknown")
        palette    = analysis.get("color_palette", [])
        furn_types = analysis.get("furniture_types", [])
        materials  = analysis.get("materials", [])
        mood       = analysis.get("lighting_mood", "")
        layout     = analysis.get("layout_features", "")
        decor      = analysis.get("decor_items", [])

        # 2) 스타일 설명 텍스트 + 임베딩
        t1 = time.time()
        parts = [
            f"style: {room_style}",
            f"colors: {', '.join(palette)}",
            f"furniture: {', '.join(furn_types)}",
            f"materials: {', '.join(materials)}",
            f"mood: {mood}",
            f"layout: {layout}",
            f"decor: {', '.join(decor)}"
        ]
        if style_keywords:
            parts.append(f"keywords: {', '.join(style_keywords)}")
        style_desc = " | ".join(parts)
        print("2) 스타일 설명 텍스트:", style_desc)

        embedding = model_manager.text_model.encode(
            [style_desc],
            normalize_embeddings=True
        )  # (1, dim), already normalized
        print(f"2) 임베딩 생성 완료: {time.time() - t1:.2f}s, shape: {embedding.shape}")

        # 3) 벡터 인덱스 검색 및 가격 필터링
        t2 = time.time()
        if vector_index.index is None:
            raise HTTPException(status_code=503, detail="Vector index 준비 중입니다. 잠시 후 다시 시도하세요.")

        # raw hits를 넉넉히 받아서 가격 필터 후 top_k로 재정렬
        raw_hits = vector_index.query(embedding.astype("float32"), top_k=top_k * 2)
        filtered_hits = []
        for doc, score in raw_hits:
            # 가격 파싱
            try:
                price = int(str(doc.get("price", "0")).replace(",", "").strip())
            except ValueError:
                continue
            # 가격 조건 검사
            if (min_price is not None and price < min_price) or \
               (max_price is not None and price > max_price):
                continue
            filtered_hits.append((doc, score))
            if len(filtered_hits) >= top_k:
                break

        print(f"3) 벡터 검색 및 필터링 완료: {time.time() - t2:.2f}s, 반환 개수: {len(filtered_hits)}")

        if not filtered_hits:
            return [{"이름": "추천 실패", "추천이유": "조건에 맞는 제품이 없습니다."}]

        # 4) 결과 포맷
        t3 = time.time()
        results = []
        for doc, score in filtered_hits:
            results.append({
                "이름":     doc.get("name", ""),
                "설명":     doc.get("description", ""),
                "링크":     doc.get("link", ""),
                "이미지":   doc.get("imageUrl", ""),
                "가격":     doc.get("price", ""),
                "추천이유": f"{room_style} 스타일과 유사도 {score:.3f}"
            })
        print(f"4) 결과 포맷 완료: {time.time() - t3:.2f}s, 개수: {len(results)}")
        print(f"전체 파이프라인 소요: {time.time() - t0:.2f}s")

        return results

    except HTTPException:
        # 이미 HTTPException이면 그대로 전달
        raise
    except Exception as e:
        print("❌ 가구 추천 오류:", e)
        raise HTTPException(status_code=500, detail=f"추천 오류: {e}")
