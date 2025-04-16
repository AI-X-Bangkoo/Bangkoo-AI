import torch
import os

GROUNDING_DINO_CONFIG = os.path.join("download_models", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_WEIGHTS = os.path.join("download_models", "groundingdino_swint_ogc.pth")
GROUNDING_TEXT_PROMPT = {
    "chair": ["chair", "stool", "armchair", "reclining chair"],
    "appliance": ["induction", "dishwasher", "microwave", "oven", "refrigerator with freezer", "freezer fridge", "refrigerator", "freezer"],
    "table": ["dining table", "table", "dining", "bar table", "adjustable desk", "desk with drawers", "desk", "gaming desk", "desk combination", "standing desk frame"],
    "bed/mattress": ["bed", "mattress", "bed frame", "headboard"],
    "bedding/pillow": ["blanket", "pillow", "bedspread"],
    "sofa": ["sofa", "couch", "recliner", "sofa bed"],
    "cushion/throw": ["cushion", "throw blanket"],
    "storage": ["storage", "drawer", "cabinet", "shelf", "bookcase", "wardrobe", "console", "storage unit", "storage box", "basket", "bag", "rolling container", "wine rack", "food container"],
    "outdoor": ["outdoor", "garden", "picnic", "bench", "exterior"],
    "plant/pot": ["plant pot", "planter", "vase", "flower vase", "plant"],
    "decor": ["frame", "decor", "object", "candle", "artificial flower", "stand", "candle dish", "candle holder", "lantern", "scented candle", "horse ornament"],
    "rug": ["rug", "carpet", "mat", "flatweave rug", "high pile rug", "sheepskin rug", "low pile rug"],
    "lighting": ["lamp", "light", "stand light", "chandelier"],
    "cover": ["bed frame cover", "cover"],
    "tray": ["laptop tray", "bed tray", "tray"]
    }
GROUNDING_BOX_THRESHOLD = 0.3
GROUNDING_TEXT_THRESHOLD = 0.25
SAM2_CHECKPOINT = os.path.join("download_models", "sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG = os.path.join("download_models", "sam2.1_hiera_large.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
