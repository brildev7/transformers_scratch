"""
추론용 한국어 소형 언어모델 클래스
"""

import torch
import torch.nn as nn
import json
import os
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """추론용 모델 설정"""
    vocab_size: int = 65536
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 32
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    @classmethod
    def from_json(cls, json_path: str):
        """JSON 파일에서 설정 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 키를 클래스 필드에 매핑
        config_data = {
            'vocab_size': data.get('vocab_size', 65536),
            'hidden_size': data.get('hidden_size', 2048),
            'num_layers': data.get('num_layers', 24),
            'num_heads': data.get('num_heads', 32),
            'intermediate_size': data.get('intermediate_size', 8192),
            'max_position_embeddings': data.get('max_position_embeddings', 4096)
        }
        
        return cls(**config_data)


class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Q, K, V 계산
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 스코어 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 인과적 마스크 적용 (자기회귀 생성을 위해)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        
        # 패딩 마스크 적용 (선택적)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # 소프트맥스 적용
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 가중합 계산
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """피드포워드 네트워크"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """트랜스포머 블록"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        self.feed_forward = FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
        
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm 구조
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class InferenceModel(nn.Module):
    """추론용 한국어 소형 언어모델"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        
        # 임베딩 레이어
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # 트랜스포머 블록들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # 최종 레이어 정규화
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 언어 모델링 헤드
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 가중치 타이잉 (임베딩과 출력 레이어 가중치 공유)
        self.lm_head.weight = self.token_embedding.weight
        
        print(f"모델 로드 완료 - 파라미터 수: {self.get_num_params():,}")
    
    def get_num_params(self) -> int:
        """모델 파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (선택적)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # 위치 인덱스 생성
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 임베딩
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        
        # 트랜스포머 블록들 통과
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # 최종 레이어 정규화
        x = self.ln_final(x)
        
        # 언어 모델링 헤드
        logits = self.lm_head(x)
        
        return logits
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "auto") -> "InferenceModel":
        """
        체크포인트에서 모델 로드
        
        Args:
            checkpoint_path: 체크포인트 디렉토리 경로
            device: 모델을 로드할 디바이스 ("auto", "cpu", "cuda")
        
        Returns:
            로드된 모델 인스턴스
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"체크포인트에서 모델 로드 중: {checkpoint_path}")
        print(f"사용 디바이스: {device}")
        
        # 설정 파일 로드
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        
        config = InferenceConfig.from_json(config_path)
        
        # 모델 초기화
        model = cls(config)
        
        # 모델 가중치 로드
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        print("모델 가중치 로드 중...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 체크포인트에서 model state dict 추출
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 키 이름 호환성 처리
        new_state_dict = {}
        for key, value in state_dict.items():
            # 'module.' 접두사 제거 (DDP 사용 시)
            if key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()  # 추론 모드로 설정
        
        print("모델 로드 완료!")
        return model
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 pad_token_id: int = 0) -> torch.Tensor:
        """
        자기회귀적 텍스트 생성
        
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_len]
            max_length: 최대 생성 길이
            temperature: 샘플링 온도 (높을수록 다양한 출력)
            top_k: top-k 샘플링
            top_p: nucleus 샘플링
            do_sample: 샘플링 여부 (False면 greedy)
            pad_token_id: 패딩 토큰 ID
        
        Returns:
            생성된 토큰 ID [batch_size, generated_length]
        """
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 생성된 시퀀스 초기화
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 현재 시퀀스에 대한 로짓 계산
                logits = self.forward(generated)
                
                # 마지막 토큰의 로짓만 사용
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Top-k 필터링
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Top-p (nucleus) 필터링
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # cumulative_probs > top_p인 토큰들 제거
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # 샘플링
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy 디코딩
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 생성된 토큰을 시퀀스에 추가
                generated = torch.cat([generated, next_token], dim=1)
                
                # 최대 위치 임베딩 길이 체크
                if generated.size(1) >= self.config.max_position_embeddings:
                    break
        
        return generated 