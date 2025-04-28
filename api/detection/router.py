from fastapi import UploadFile, File, HTTPException, APIRouter,Form
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from utils.dino_sam2_config import (
    GROUNDING_BOX_THRESHOLD, GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS,
    GROUNDING_TEXT_PROMPT, SAM2_CHECKPOINT, SAM2_MODEL_CONFIG, DEVICE, GROUNDING_TEXT_THRESHOLD
)
from datetime import datetime
from io import BytesIO
import base64
from typing import Annotated
import sys
import os
import torch
from api.detection.groundingdino.groundingdino.util.inference import load_model, predict
from api.detection.sam2.build_sam import build_sam2
from api.detection.sam2.sam2_image_predictor import SAM2ImagePredictor
router = APIRouter()


sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS,DEVICE)
# try:
#     from api.detection.groundingdino.groundingdino.util.inference import load_model, predict
#     from api.detection.sam2.build_sam import build_sam2
#     from api.detection.sam2.sam2_image_predictor import SAM2ImagePredictor
#     print("GroundingDINO & SAM2 installed.")
#     GROUNDINGDINO_AVAILABLE = True
# except ImportError:
#     print("Warning: GroundingDINO or SAM2 not installed. Object detection features will be disabled.")
#     GROUNDINGDINO_AVAILABLE = False

def image_to_base64(pil_img: Image.Image, format="PNG") -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    return f"data:image/{format.lower()};base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

def resize_image(image: Image, target_width: int, target_height: int) -> Image:
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def match_category(label, prompt_dict):
    label_lower = label.lower()
    for category, keywords in prompt_dict.items():
        for keyword in keywords:
            if keyword.lower() in label_lower:
                return category
    return "unknown"




@router.post("/detect_all_base64")
async def detect_all_base64(
    file: UploadFile = File(...),
    canvasWidth: int = Form(...),
    canvasHeight: int = Form(...)
    ):
    # if not GROUNDINGDINO_AVAILABLE:
    #     raise HTTPException(
    #         status_code=501,
    #         detail="Object detection is not available. GroundingDINO package is not installed."
    #     )
    start = datetime.now()
    print("파일 읽기 시도")
    print("📐 canvas:", canvasWidth, canvasHeight)
    image = Image.open(BytesIO(await file.read())).convert("RGB")
   
    image = resize_image(image, canvasWidth, canvasHeight)
    image_np = np.array(image)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(DEVICE)
    prompt_list = sum(GROUNDING_TEXT_PROMPT.values(), [])
    caption = ". ".join(prompt_list)

    boxes, logits, phrases = predict(
        model=grounding_model,
        image=image_tensor,
        caption=caption,
        box_threshold=GROUNDING_BOX_THRESHOLD,
        text_threshold=GROUNDING_TEXT_THRESHOLD,
        device=DEVICE
    )

    if boxes.shape[0] == 0:
        raise HTTPException(status_code=404, detail="탐지된 객체가 없습니다")

    boxes = boxes * torch.tensor([image.width, image.height, image.width, image.height], device=boxes.device)
    boxes = box_cxcywh_to_xyxy(boxes)
    input_boxes = boxes.cpu().numpy()

    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(box=input_boxes, multimask_output=False)

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    label_to_category = {
        synonym.lower(): category
        for category, synonyms in GROUNDING_TEXT_PROMPT.items()
        for synonym in synonyms
    }

    results = []
    thumbnails = []
    base_image = image_np.copy()
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
               (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]

    for idx, (mask, box, label) in enumerate(zip(masks, input_boxes, phrases)):
        label_clean = label.strip().lower()
        category = label_to_category.get(label_clean, "unknown")
        ys, xs = np.where(mask > 0)
        if len(xs) < 5 or len(ys) < 5:
            continue

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        cropped_mask = mask[min_y:max_y + 1, min_x:max_x + 1]

        result = {
            "class": match_category(label.strip(), GROUNDING_TEXT_PROMPT),
            "score": 1.0,
            "bbox": [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)],
            "mask": cropped_mask.astype(int).tolist()
        }
        bbox = [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]
        print(f"[{idx}] 감지 결과: {match_category(label.strip(), GROUNDING_TEXT_PROMPT)}, bbox: {bbox}")
        results.append(result)

        rgba_img = image.crop((min_x, min_y, max_x + 1, max_y + 1)).convert("RGBA")
        mask_img = Image.fromarray((cropped_mask * 255).astype(np.uint8)).resize(rgba_img.size)
        rgba_img.putalpha(mask_img)
        thumbnails.append(image_to_base64(rgba_img))

        color = palette[idx % len(palette)]
        full_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        full_mask[min_y:max_y + 1, min_x:max_x + 1] = cropped_mask
        for c in range(3):
            base_image[:, :, c] = np.where(
                full_mask == 1,
                (0.6 * base_image[:, :, c] + 0.4 * color[c]).astype(np.uint8),
                base_image[:, :, c]
            )

    final_base64 = image_to_base64(Image.fromarray(base_image), format="JPEG")
    print("탐지된 객체 수:", len(results), "/ 실행 시간:", datetime.now() - start)

    return {
        "results": results,
        "final_image_base64": final_base64,
        "thumbnails_base64": thumbnails,
        "original_image_base64": image_base64  # ✅ 새로 추가
    }
