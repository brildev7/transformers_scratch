"""
학습 코드와 호환되는 추론용 모델
train_h100_dual.py의 SimpleTransformerModel과 동일한 구조
"""

import torch
import torch.nn as nn
import json
import os
from typing import Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingCompatibleConfig:
    """학습 호환 설정 (train_h100_dual.py와 동일)"""
    # 모델 설정 - 학습 코드와 동일
    model_name: str = "korean-sllm-h100"
    vocab_size: int = 65536
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 32
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096
    
    # 추론 설정
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    @classmethod
    def from_json(cls, json_path: str):
        """JSON 파일에서 설정 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 학습 설정을 추론 설정으로 매핑
        config = cls()
        if 'vocab_size' in data:
            config.vocab_size = data['vocab_size']
        if 'hidden_size' in data:
            config.hidden_size = data['hidden_size']
        if 'num_layers' in data:
            config.num_layers = data['num_layers']
        if 'num_heads' in data:
            config.num_heads = data['num_heads']
        if 'intermediate_size' in data:
            config.intermediate_size = data['intermediate_size']
        if 'max_position_embeddings' in data:
            config.max_position_embeddings = data['max_position_embeddings']
            
        return config


class TrainingCompatibleModel(nn.Module):
    """학습용 SimpleTransformerModel과 동일한 구조"""
    
    def __init__(self, config: TrainingCompatibleConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers - 학습 코드와 동일
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # weight tying - 학습 코드와 동일
        self.head.weight = self.embeddings.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """가중치 초기화 - 학습 코드와 동일"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass - 학습 코드와 동일"""
        batch_size, seq_len = input_ids.shape
        
        # Position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Attention mask for transformer
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # invert for transformer
        
        # Transformer
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = self.ln_f(hidden_states)
        
        # Language model head
        logits = self.head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {"loss": loss, "logits": logits}
    
    def generate(self, input_ids, attention_mask=None, max_length=100, 
                 temperature=1.0, top_k=50, top_p=0.9, do_sample=True,
                 pad_token_id=0, eos_token_id=2):
        """텍스트 생성"""
        print(f"🚀 Generate 호출: max_length={max_length}, input shape={input_ids.shape}")
        
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 생성 과정에서 사용할 변수들
        generated = input_ids.clone()
        past_length = input_ids.size(1)
        
        print(f"🔍 시작: batch_size={batch_size}, past_length={past_length}, max_length={max_length}")
        
        # 생성할 토큰 수 계산
        tokens_to_generate = max_length - past_length
        if tokens_to_generate <= 0:
            print(f"⚠️ 생성할 토큰이 없음: 현재 길이 {past_length} >= max_length {max_length}")
            return generated
        
        with torch.no_grad():
            for step in range(tokens_to_generate):
                print(f"🔄 Step {step}: 현재 길이={generated.size(1)}")
                
                if step >= tokens_to_generate:
                    print(f"🔚 최대 길이 도달: {step}/{tokens_to_generate}")
                    break
                    
                try:
                    # Forward pass
                    outputs = self.forward(generated, attention_mask=attention_mask)
                    logits = outputs["logits"]
                    
                    # 마지막 토큰의 로짓
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Top-k 필터링 (안전한 방식)
                    if top_k > 0 and top_k < next_token_logits.size(-1):
                        try:
                            values, indices = torch.topk(next_token_logits, k=top_k, dim=-1)
                            min_values = values[:, -1:].expand_as(next_token_logits)
                            indices_to_remove = next_token_logits < min_values
                            next_token_logits[indices_to_remove] = float('-inf')
                        except Exception as e:
                            print(f"❌ Top-k 필터링 오류: {e}")
                    
                    # Top-p 필터링 (안전한 방식)
                    if top_p < 1.0:
                        try:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                            sorted_indices_to_remove[:, 0] = False
                            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = float('-inf')
                        except Exception as e:
                            print(f"❌ Top-p 필터링 오류: {e}")
                    
                    # 샘플링
                    try:
                        if do_sample:
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        # 생성된 토큰 추가
                        generated = torch.cat([generated, next_token], dim=1)
                        token_value = next_token[0, 0].item()
                        
                        if step % 10 == 0:  # 10스텝마다만 출력
                            print(f"🔄 Step {step}: 토큰 {token_value} 생성, 길이 {generated.size(1)}")
                        
                        # EOS 토큰이 생성되면 중단
                        if token_value == eos_token_id:
                            print(f"🔚 EOS 토큰 감지, 생성 중단 (길이: {generated.size(1)})")
                            break
                            
                        # attention_mask 업데이트
                        if attention_mask is not None:
                            attention_mask = torch.cat([
                                attention_mask, 
                                torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
                            ], dim=1)
                            
                    except Exception as e:
                        print(f"❌ 샘플링 오류: {e}")
                        break
                        
                        # EOS 토큰이 생성되면 중단
                        if token_value == eos_token_id:
                            print(f"🔚 EOS 토큰 감지, 생성 중단 (길이: {generated.size(1)})")
                            break
                            
                        # attention_mask 업데이트
                        if attention_mask is not None:
                            attention_mask = torch.cat([
                                attention_mask, 
                                torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
                            ], dim=1)
                            
                    except Exception as e:
                        print(f"❌ 샘플링 오류: {e}")
                        break
                        
                except Exception as e:
                    print(f"❌ Generation step {step} 오류: {e}")
                    print(f"   generated shape: {generated.shape}")
                    print(f"   attention_mask: {attention_mask.shape if attention_mask is not None else None}")
                    import traceback
                    traceback.print_exc()
                    break
        
        return generated
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "auto"):
        """체크포인트에서 모델 로드"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_path = Path(model_path)
        
        # 설정 파일 로드
        config_file = model_path / "config.json"
        if config_file.exists():
            config = TrainingCompatibleConfig.from_json(str(config_file))
        else:
            print("⚠️ config.json이 없습니다. 기본 설정을 사용합니다.")
            config = TrainingCompatibleConfig()
        
        # 모델 초기화
        model = cls(config)
        
        # 가중치 로드
        model_file = model_path / "pytorch_model.bin"
        if model_file.exists():
            print(f"체크포인트에서 모델 로드 중: {model_path}")
            print(f"사용 디바이스: {device}")
            
            checkpoint = torch.load(model_file, map_location=device)
            
            # 체크포인트가 래핑된 경우 (예: DDP)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 키 정리: _orig_mod. 접두사 제거 (torch.compile 사용시)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                # _orig_mod. 접두사 제거
                if key.startswith("_orig_mod."):
                    clean_key = key.replace("_orig_mod.", "")
                    cleaned_state_dict[clean_key] = value
                # module. 접두사 제거 (DDP 사용시)
                elif key.startswith("module."):
                    clean_key = key.replace("module.", "")
                    cleaned_state_dict[clean_key] = value
                else:
                    cleaned_state_dict[key] = value
            
            print(f"🔧 키 정리 완료: {len(state_dict)} -> {len(cleaned_state_dict)}")
            print(f"정리된 키 샘플: {list(cleaned_state_dict.keys())[:3]}")
            
            model.load_state_dict(cleaned_state_dict, strict=False)
            print("모델 로드 완료!")
            
            # 모델 정보 출력
            total_params = sum(p.numel() for p in model.parameters())
            print(f"모델 로드 완료 - 파라미터 수: {total_params:,}")
            
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_file}")
            
        return model.to(device)
    
    def get_model_info(self):
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "model_name": self.config.model_name,
            "model_parameters": total_params,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "max_position_embeddings": self.config.max_position_embeddings,
            "device": str(next(self.parameters()).device),
        } 