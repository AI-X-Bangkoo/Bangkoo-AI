import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    def __init__(self, img_dim, txt_dim, output_dim):
        """
        이미지 임베딩과 텍스트 임베딩을 결합하는 Fusion 네트워크를 초기화
        
        Args:
            img_dim (int): 이미지 임베딩의 차원 (예: 1024)
            txt_dim (int): 텍스트 임베딩의 차원 (예: 768)
            output_dim (int): 최종 결합 임베딩의 차원 (예: 1792)
        """
        super(FusionNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + txt_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, img_embed, txt_embed):
        """
        이미지 및 텍스트 임베딩을 결합하여 최종 Fusion 임베딩을 출력
        
        Args:
            img_embed (torch.Tensor): 이미지 임베딩, shape (batch_size, img_dim)
            txt_embed (torch.Tensor): 텍스트 임베딩, shape (batch_size, txt_dim)
            
        Returns:
            torch.Tensor: 결합 임베딩, shape (batch_size, output_dim), L2 정규화됨
        """
        # 두 임베딩을 연결
        x = torch.cat([img_embed, txt_embed], dim=1)
        # 정의된 fully-connected 네트워크를 통과
        x = self.fc(x)
        # L2 노름으로 정규화하여 출력
        x = x / x.norm(dim=1, keepdim=True)
        return x

if __name__ == "__main__":
    # 테스트용: 임의의 더미 입력 데이터를 생성하여 네트워크의 동작을 확인
    fusion_net = FusionNetwork(1024, 768, 1792)
    print("[DEBUG] Fusion 네트워크 생성 완료")
    
    # 배치 크기 1의 더미 데이터 생성
    dummy_img_embedding = torch.randn(1, 1024)
    dummy_txt_embedding = torch.randn(1, 768)
    
    # Forward pass 수행
    fused_output = fusion_net(dummy_img_embedding, dummy_txt_embedding)
    print("[DEBUG] Fusion 네트워크 출력 형태:", fused_output.shape)
    print("[DEBUG] Fusion 네트워크 출력 벡터 L2 norm:", fused_output.norm(dim=1))
