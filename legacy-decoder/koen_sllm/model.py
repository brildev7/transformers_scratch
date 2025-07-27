"""
Korean sLLM Transformer Model Implementation
한국어 소형 언어모델 트랜스포머 구현
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    """위치 인코딩 클래스"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 위치 인코딩 행렬 생성
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # sin, cos 함수를 이용한 위치 인코딩
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # batch dimension 추가
        
        # buffer로 등록 (학습되지 않는 파라미터)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션 메커니즘"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query, Key, Value 변환 레이어
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 캐시된 마스크를 위한 buffer
        self.register_buffer('causal_mask', None)
    
    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """인과적 마스크 생성 (상삼각 마스크)"""
        if (self.causal_mask is None or 
            self.causal_mask.size(0) < seq_len):
            # 새로운 마스크 생성
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.bool()
            self.register_buffer('causal_mask', mask)
        
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] 패딩 마스크 (선택적)
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # Q, K, V 계산
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # 멀티헤드를 위해 reshape
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # [batch_size, n_heads, seq_len, head_dim]
        
        # 어텐션 점수 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # [batch_size, n_heads, seq_len, seq_len]
        
        # 인과적 마스크 적용 (디코더)
        causal_mask = self.get_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 패딩 마스크 적용 (선택적)
        if mask is not None:
            # mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스 적용
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 어텐션 적용
        out = torch.matmul(attn_weights, v)
        # [batch_size, n_heads, seq_len, head_dim]
        
        # 헤드들을 다시 합치기
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 출력 변환
        out = self.w_o(out)
        
        return out


class FeedForward(nn.Module):
    """피드포워드 네트워크"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GPT에서 사용하는 활성화 함수
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """트랜스포머 블록 (레이어)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        self.feed_forward = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        
        # 레이어 정규화
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] 패딩 마스크
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 어텐션 + 잔차 연결
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = x + residual
        
        # 피드포워드 + 잔차 연결
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class KoreanSLLM(nn.Module):
    """한국어 소형 언어모델"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 임베딩 레이어
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        
        # 트랜스포머 블록들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 최종 레이어 정규화
        self.ln_final = nn.LayerNorm(config.d_model)
        
        # 언어 모델링 헤드
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
        # 임베딩과 출력 레이어 가중치 공유 (선택적)
        self.lm_head.weight = self.token_embedding.weight
        
        print(f"모델 파라미터 수: {self.get_num_params():,}")
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_num_params(self) -> int:
        """모델 파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def create_padding_mask(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """패딩 마스크 생성"""
        return (input_ids != pad_token_id).long()
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (선택적)
            labels: [batch_size, seq_len] 다음 토큰 예측을 위한 라벨 (선택적)
        
        Returns:
            Dict containing logits and optionally loss
        """
        batch_size, seq_len = input_ids.size()
        
        # 패딩 마스크 생성
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        
        # 토큰 임베딩
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # 위치 인코딩 추가
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 트랜스포머 블록들 통과
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # 최종 레이어 정규화
        x = self.ln_final(x)
        
        # 언어 모델링 헤드
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        outputs = {'logits': logits}
        
        # 손실 계산 (학습 시)
        if labels is not None:
            # 다음 토큰 예측을 위해 라벨을 한 토큰씩 밀기
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 토큰 무시
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            outputs['loss'] = loss
        
        return outputs
    
    def generate(self, input_ids: torch.Tensor, 
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 pad_token_id: int = 0,
                 eos_token_id: int = 3) -> torch.Tensor:
        """
        텍스트 생성 함수
        
        Args:
            input_ids: [batch_size, seq_len] 입력 토큰
            max_length: 최대 생성 길이
            temperature: 샘플링 온도
            top_k: top-k 샘플링
            top_p: nucleus 샘플링
            pad_token_id: 패딩 토큰 ID
            eos_token_id: 종료 토큰 ID
        
        Returns:
            [batch_size, generated_seq_len] 생성된 토큰
        """
        self.eval()
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        
        # 결과를 저장할 텐서
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # 현재까지의 시퀀스로 다음 토큰 예측
                outputs = self.forward(generated)
                logits = outputs['logits']
                
                # 마지막 토큰의 logits만 사용
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k 필터링
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p 필터링 (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # top-p를 초과하는 토큰들 제거
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 다음 토큰 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 생성된 토큰을 시퀀스에 추가
                generated = torch.cat([generated, next_token], dim=1)
                
                # EOS 토큰이 생성되면 종료
                if next_token.item() == eos_token_id:
                    break
        
        return generated


def count_parameters(model: nn.Module) -> int:
    """모델의 학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 모델 테스트
    from .config import ModelConfig
    
    config = ModelConfig(
        vocab_size=32000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=2048
    )
    
    model = KoreanSLLM(config)
    
    # 더미 입력으로 테스트
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"입력 형태: {input_ids.shape}")
    
    # 순전파 테스트
    outputs = model(input_ids)
    print(f"출력 logits 형태: {outputs['logits'].shape}")
    
    # 생성 테스트
    generated = model.generate(input_ids[:1], max_length=20)
    print(f"생성된 시퀀스 형태: {generated.shape}") 