# import torch

# print("[TEST] torch version:", torch.__version__)
# print("[TEST] torch available:", torch.cuda.is_available())
# print("[TEST] done")
# torch_test2.py
from model_loader import get_model_manager

print("Model loader import 완료")

model_manager = get_model_manager()
print("Model manager 생성 완료")
