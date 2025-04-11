import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer

"""
    최초 작성자: 김동규
    최초 작성일: 2025-04-07
    
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
        print("[8] All models loaded")
        
        self.ready = True
        
model_manager = ModelManager()

