import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import requests
import os
import numpy as np
import pymongo
from pymongo import MongoClient

"""
    최초 작성자: 김동규
    최초 작성일: 2025-04-07
    수정일: 2025-04-11 (김범석) (sam2,dino model 추가)
    모델 및 DB 초기화를 lazy-load 또는 startup 이벤트에서 처리
    수정일: 2025-04-18 (김병훈)
    새로 추가돼는 데이터의 이미지, 텍스트 임베딩 작업 수정 
"""



class ModelManager:
    def __init__(self):
        # PyTorch 버전 및 CUDA 가능 여부 출력
        print(torch.__version__)
        print(torch.cuda.is_available())
        
        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[DEBUG] self.device: ", self.device)
        
        # 모델 관련 변수 초기화
        self.clip_model = None
        self.clip_processor = None
        self.text_model = None
        self.text_tokenizer = None
        
        # Grounding DINO & SAM2 경로 변수
        self.dino_model = None
        self.dino_pth = None
        self.sam2_model = None
        self.sam2_yaml = None

        self.ready = False  # 준비 완료 상태 표시

        # MongoDB 연결
        self.mongo_client = None
        self.db = None
        self.collection = None

    def load(self):
        # 1. CLIP 모델 로드
        print("[1] Loading CLIP model...")
        self.clip_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)

        # 2. 필요하지 않은 설정 제거
        if hasattr(self.clip_model.config, 'use_flash_attention'):
            self.clip_model.config.use_flash_attention = False

        # 🔥 xFormers 관련 설정 제거 (안 씀)
        if hasattr(self.clip_model.config, 'use_xformers'):
            del self.clip_model.config.use_xformers  # 완전히 삭제

        # 3. 디바이스로 이동 및 평가 모드 설정
        print("[2] Moving to device...")
        self.clip_model = self.clip_model.to(self.device).eval().to(torch.float32)
        print("[3] CLIP model ready.")

        # 4. 이미지 전처리 Processor 로드
        print("[4] Loading processor...")
        self.clip_processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", use_fast=True, trust_remote_code=True)
        print("[5] Processor ready.")

        # 5. 텍스트 임베딩 모델 로드
        print("[6] Loading SentenceTransformer...")
        self.text_model = SentenceTransformer("intfloat/e5-base-v2", device=self.device)

        # 6. 텍스트 토크나이저 로드
        print("[7] Loading AutoTokenizer...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")  

        # 7. 객체 탐지/분할 관련 모델 파일 다운로드
        print("[8] Loading Grounding Dino & Sam2 model...")
        download_dir = "download_models"
        os.makedirs(download_dir, exist_ok=True)

        files = {
            "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            "sam2.1_hiera_large.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            "GroundingDINO_SwinT_OGC.py": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        }

        for filename, url in files.items():
            file_path = os.path.join(download_dir, filename)
            if os.path.exists(file_path):
                continue

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

        # MongoDB 연결 설정
        self.connect_to_mongo()

    def connect_to_mongo(self):
        """MongoDB 연결 및 DB/컬렉션 설정"""
        try:
            print("[DEBUG] MongoDB 연결 시도...")
            self.mongo_client = MongoClient("mongodb://localhost:27017/")  # MongoDB 연결 URL
            self.db = self.mongo_client["my_database"]
            self.collection = self.db["my_collection"]
            print("[DEBUG] MongoDB 연결 성공!")
        except Exception as e:
            print(f"[오류] MongoDB 연결 실패: {e}")

    def encode_image_from_url(self, image_url):
        """
        이미지 URL을 받아 CLIP 임베딩 생성
        :param image_url: str
        :return: numpy float32 array (1024,)
        """
        try:
            response = requests.get(image_url)
            print(f"[DEBUG] 이미지 URL 확인: {image_url}")

            if response.status_code != 200:
                print(f"[오류] 이미지 요청 실패: {response.status_code} - {image_url}")
                return None

            img = Image.open(BytesIO(response.content)).convert("RGB")
            print(f"[DEBUG] 이미지 로드 성공: {image_url}")

            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_embedding = self.clip_model.get_image_features(**inputs)
                image_embedding = image_embedding.squeeze(0)

            if image_embedding is None:
                print(f"[오류] 이미지 임베딩 결과가 None입니다.")
                return None

            print(f"[DEBUG] 이미지 임베딩 완료. shape: {image_embedding.shape}")
            return image_embedding.cpu().numpy().astype(np.float32)

        except Exception as e:
            print(f"[오류] 이미지 임베딩 실패: {e}")
            return None

    def encode_text(self, text: str):
        """
        텍스트를 받아 임베딩을 생성하고 numpy float32 배열로 반환
        :param text: str
        :return: numpy array (768,) - float32
        """
        try:
            if not text or not isinstance(text, str):
                print("[오류] 텍스트가 비어있거나 문자열이 아님")
                return None

            print(f"[DEBUG] 텍스트 임베딩 시작: {text}")
            embedding = self.text_model.encode(
                [f"query: {text}"],
                normalize_embeddings=True
            )[0].astype(np.float32)
            
            print(f"[DEBUG] 텍스트 임베딩 완료. shape: {embedding.shape}")
            return embedding

        except Exception as e:
            print(f"[오류] 텍스트 임베딩 실패: {e}")
            return None

    def save_to_mongo(self, data):
        """
        MongoDB에 데이터를 저장하는 함수
        :param data: dict 형태의 데이터
        :return: 저장된 데이터의 ID
        """
        try:
            print("[DEBUG] MongoDB에 데이터 저장 중...")
            result = self.collection.insert_one(data)
            print(f"[DEBUG] 데이터 저장 완료. 저장된 _id: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"[오류] MongoDB 저장 실패: {e}")
            return None
    

# 클래스 인스턴스 생성
model_manager = ModelManager()

# 예시: 텍스트 임베딩 후 MongoDB에 저장
text = "This is an example text for embedding."
embedding = model_manager.encode_text(text)

if embedding is not None:
    data = {
        "text": text,
        "embedding": embedding.tolist()  # numpy array를 리스트로 변환하여 저장
    }
    model_manager.save_to_mongo(data)
