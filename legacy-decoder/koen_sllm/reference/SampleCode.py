import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention 모듈입니다.
    GPT-Neo와 같은 디코더-온리 모델의 핵심 구성 요소로,
    각 토큰이 시퀀스 내 이전 토큰들에만 주의(attend)를 기울이도록 합니다.
    """
    def __init__(self, config):
        super().__init__()
        assert config['hidden_size'] % config['num_heads'] == 0
        # Q, K, V를 한 번에 계산하기 위한 선형 계층
        self.c_attn = nn.Linear(config['hidden_size'], 3 * config['hidden_size'])
        # 출력 프로젝션을 위한 선형 계층
        self.c_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        # 드롭아웃
        self.attn_dropout = nn.Dropout(config['attention_dropout'])
        self.resid_dropout = nn.Dropout(config['resid_dropout'])
        # 하이퍼파라미터 저장
        self.n_head = config['num_heads']
        self.n_embd = config['hidden_size']
        # 인과적 마스크(causal mask)를 버퍼로 등록
        # 하삼각행렬(lower triangular matrix)을 사용하여 미래 토큰을 마스킹합니다.
        self.register_buffer("bias", torch.tril(torch.ones(config['max_position_embeddings'], config['max_position_embeddings']))
                                       .view(1, 1, config['max_position_embeddings'], config['max_position_embeddings']))

    def forward(self, x):
        B, T, C = x.size() # 배치 크기, 시퀀스 길이, 임베딩 차원

        # Q, K, V 계산 및 multi-head를 위해 분할
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # head들을 다시 합침

        # 출력 프로젝션
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    Transformer 블록 내의 Feed-Forward Network (FFN)입니다.
    GPT-Neo에서는 GELU 활성화 함수를 사용합니다.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['hidden_size'], 4 * config['hidden_size'])
        self.gelu    = nn.GELU(approximate='tanh') # 'gelu_new'와 유사
        self.c_proj  = nn.Linear(4 * config['hidden_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['resid_dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    하나의 Transformer 블록입니다.
    Causal Self-Attention과 MLP를 포함하며, 각각의 앞뒤에 Layer Normalization과
    Residual Connection이 적용됩니다.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_epsilon'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_epsilon'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTNeo(nn.Module):
    """
    GPT-Neo 1.3B 모델의 전체 아키텍처입니다.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['hidden_size']),
            wpe = nn.Embedding(config['max_position_embeddings'], config['hidden_size']),
            drop = nn.Dropout(config['embed_dropout']),
            h = nn.ModuleList()]),
            ln_f = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_epsilon']),
        ))
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight # 가중치 공유

        # 가중치 초기화
        self.apply(self._init_weights)
        # 모든 잔차 경로에 특별한 스케일링 적용
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config['initializer_range']/math.sqrt(2 * config['num_layers']))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['initializer_range'])

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config['max_position_embeddings'], f"Cannot forward sequence of length {t}, block size is only {self.config['max_position_embeddings']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 학습 중인 경우, loss 계산
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 추론 시, 마지막 토큰의 로짓만 계산하여 효율성 증대
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding=True):
        """
        모델의 파라미터 수를 반환합니다.
        non_embedding=True일 경우, 토큰 임베딩 가중치를 제외하고 계산합니다.
        (가중치 공유로 인해 중복 계산될 수 있음)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


if __name__ == '__main__':
    # EleutherAI/gpt-neo-1.3B 모델의 하이퍼파라미터
    config = {
        "activation_function": "gelu_new",
        "attention_dropout": 0,
        "embed_dropout": 0,
        "gradient_checkpointing": False,
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "max_position_embeddings": 2048,
        "model_type": "gpt_neo",
        "num_heads": 16,
        "num_layers": 24,
        "resid_dropout": 0,
        "vocab_size": 50257,
    }

    # 모델 인스턴스화
    model = GPTNeo(config)
    print("모델 아키텍처:")
    print(model)

    # 파라미터 수 계산 및 출력
    num_params = model.get_num_params()
    print(f"\n총 파라미터 수: {num_params / 1e9:.3f}B")

    # 더미 입력으로 forward pass 테스트
    print("\n더미 입력 테스트:")
    dummy_input = torch.randint(0, config['vocab_size'], (2, 128)) # (batch_size, sequence_length)
    logits, loss = model(dummy_input, targets=dummy_input)
    print(f"입력 shape: {dummy_input.shape}")
    print(f"출력 로짓 shape: {logits.shape}")
    if loss is not None:
        print(f"계산된 손실 값: {loss.item()}")
    else:
        # 추론 시 (targets=None)
        logits_inference, _ = model(dummy_input)
        print(f"추론 시 출력 로짓 shape: {logits_inference.shape}")
