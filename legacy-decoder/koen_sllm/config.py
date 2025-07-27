"""
Model configuration for Korean sLLM
"""
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    # 모델 아키텍처
    vocab_size: int = 32000
    d_model: int = 768          # 임베딩 차원
    n_heads: int = 12           # 어텐션 헤드 수
    n_layers: int = 12          # 트랜스포머 레이어 수
    d_ff: int = 3072           # 피드포워드 네트워크 차원
    max_seq_len: int = 2048     # 최대 시퀀스 길이
    dropout: float = 0.1        # 드롭아웃 비율
    
    # 학습 설정
    learning_rate: float = 1e-4
    batch_size: int = 8         # H100 두 장 기준
    grad_accumulation_steps: int = 8
    max_steps: int = 100000
    warmup_steps: int = 2000
    save_steps: int = 1000
    eval_steps: int = 500
    
    # 데이터셋 설정
    train_data_path: str = "datasets/train"
    val_data_path: str = "datasets/val"
    tokenizer_path: str = "datasets/tokenizer"
    
    # 멀티 GPU 설정
    world_size: int = 2         # H100 두 장
    local_rank: int = 0
    
    # 기타 설정
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    def save(self, path: str):
        """설정을 JSON 파일로 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str):
        """JSON 파일에서 설정 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


# 기본 설정
DEFAULT_CONFIG = ModelConfig() 