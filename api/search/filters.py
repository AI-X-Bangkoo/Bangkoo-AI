import re

def apply_filters(products, price_range=None, keyword=None, style=None):
    filtered = []

    for p in products:
        # 가격 필터
        if price_range:
            try:
                price_str = p.get("할인가") or p.get("정상가") or ""
                numeric = re.sub(r"[^\d]", "", price_str)
                if not numeric:
                    continue
                price = int(numeric)
                if not (price_range[0] <= price <= price_range[1]):
                    continue
            except:
                continue

        # 키워드 필터
        if keyword:
            combined_text = f"{p.get('이름', '')} {p.get('설명', '')}"
            if keyword not in combined_text:
                continue

        # 스타일 필터
        if style and style not in p.get("상세설명", ""):
            continue

        filtered.append(p)

    return filtered
