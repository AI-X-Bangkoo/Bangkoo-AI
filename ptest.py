import torch


print(torch.__version__)  # PyTorch 버전 확인
print(torch.cuda.is_available())  # CUDA가 사용 가능한지 확인
print(torch.version.cuda)  # 현재 사용 중인 CUDA 버전 출력
