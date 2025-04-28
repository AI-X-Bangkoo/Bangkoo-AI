import base64
from .gemini_client import GeminiClient
from PIL import Image
from io import BytesIO

client = GeminiClient()

# 🎯 프롬프트 모음
PROMPTS = {
    "add": (
        "Make the object inside the red box look photorealistic and naturally integrated into the room."
    ),
    "remove": (
        "Remove the object outlined in red. "
        "Make the scene look natural and consistent after removal. "
        "Do not change anything else."
    ),
    "move": (
        "Blend the object outlined in red naturally into the room without changing the background."
    ),
}


async def process_placement(mode: str, background_file, reference_file=None):
    # 유효한 작업 모드인지 확인
    if mode not in PROMPTS:
        return {"error": f"Unsupported mode: {mode}"}

    # 배경 이미지 로딩
    bg_img = Image.open(BytesIO(await background_file.read())).convert("RGBA")

    # 작업 모드에 따라 처리 방식 분기
    if mode == "add":
        # if reference_file is None:
        #     return {"error": "Reference image required for 'add' mode"}
        #
        # ref_img = Image.open(BytesIO(await reference_file.read())).convert("RGBA")
        result_img = client.generate_image(PROMPTS["add"], bg_img)

    elif mode == "remove":
        result_img = client.generate_image(PROMPTS["remove"], bg_img)

    elif mode == "move":
        result_img = client.generate_image(PROMPTS["move"], bg_img)
    else:
        return {"error": "Invalid mode."}

    # 결과 처리 (예: 저장 or 버퍼 리턴)
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "message": "success",
        "filename": f"{mode}_result.png",
        "image_base64": img_base64
    }