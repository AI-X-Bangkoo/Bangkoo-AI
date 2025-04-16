import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer
from utils.dino_sam2_config import GROUNDING_DINO_CONFIG,GROUNDING_DINO_WEIGHTS,SAM2_CHECKPOINT,SAM2_MODEL_CONFIG
import requests
import os


"""
    최초 작성자: 김동규
    최초 작성일: 2025-04-07
    수정일: 2025-04-11 (김범석) (sam2,dino model 추가)
    모델 및 DB 초기화를 lazy-load 또는 startup 이벤트에서 처리
"""

class ModelManager:
    def __init__(self):
        print(torch.__version__)
        print(torch.cuda.is_available())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[DEBUG] self.device: ", self.device)
        self.clip_model = None
        self.clip_processor = None
        self.text_model = None
        self.ready = False
        self.dino_model = None
        self.dino_pth = None
        self.sam2_model = None
        self.sam2_yaml = None

    def load(self):
        print("[1] Loading CLIP model...")
        self.clip_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
        print("[2] Moving to device...")
        self.clip_model = self.clip_model.to(self.device).eval().to(torch.float32)
        print("[3] CLIP model ready.")

        print("[4] Loading processor...")
        self.clip_processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
        print("[5] Processor ready.")

        print("[6] Loading SentenceTransformer...")
        self.text_model = SentenceTransformer("intfloat/e5-base-v2", device=self.device)
        print("[7] Loading AutoTokenizer...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")  
        print("[8] Loading Grounding Dino & Sam2 model...")
        # file_id = '1B0UPAEb4yhmGNx6jPyxoQtnVUcU6gHw-'
        # url = f"https://drive.google.com/uc?id={file_id}"
        download_dir = "download_models"
        os.makedirs(download_dir, exist_ok=True)
        # gdown.download_folder(url,quiet=False,use_cookies=False)
        files = {
        "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2.1_hiera_large.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "GroundingDINO_SwinT_OGC.py": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        }

        for filename, url in files.items():
            file_path = os.path.join(download_dir, filename)
            # print(f"[DEBUG] 다운로드 시도: {filename}")

            if os.path.exists(file_path):
                # print(f"[DEBUG] {filename} 이미 존재합니다. 다운로드 생략.")
                continue  # 이미 있으면 스킵

            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[DEBUG] 다운로드 완료: {file_path}")
            else:
                print(f"[DEBUG] 다운로드 실패: {filename} ({response.status_code})")
        
        print("[9] All models loaded")
        self.ready = True
        
model_manager = ModelManager()

