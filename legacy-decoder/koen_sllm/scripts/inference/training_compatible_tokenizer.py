"""
학습 코드와 호환되는 토크나이저
train_h100_dual.py의 TextDataset.__getitem__에서 사용된 토크나이징 방식과 동일
"""

import json
import os
from typing import List, Dict, Optional, Union


class TrainingCompatibleTokenizer:
    """학습 시 사용된 해시 기반 토크나이저와 동일한 방식"""
    
    def __init__(self, vocab_file: Optional[str] = None):
        """
        Args:
            vocab_file: 어휘 파일 경로 (현재는 사용하지 않음, 해시 기반이므로)
        """
        # 특수 토큰 - 학습 코드와 동일
        self.pad_token = "<pad>"
        self.bos_token = "<s>"  # BOS
        self.eos_token = "</s>"  # EOS
        self.unk_token = "<unk>"
        
        # 특수 토큰 ID - 학습 코드와 동일
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        # 어휘 크기 - 학습 코드와 동일
        self.vocab_size = 65536
        
        print(f"학습 호환 토크나이저 초기화 완료")
        print(f"  • 어휘 크기: {self.vocab_size:,}")
        print(f"  • PAD: {self.pad_token_id}")
        print(f"  • BOS: {self.bos_token_id}")
        print(f"  • EOS: {self.eos_token_id}")
        print(f"  • UNK: {self.unk_token_id}")
        
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할 - 학습 코드와 동일"""
        return text.split()
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """토큰을 ID로 변환 - 학습 코드와 동일한 해시 방식"""
        token_ids = []
        for token in tokens:
            # 특수 토큰 처리
            if token == self.pad_token:
                token_ids.append(self.pad_token_id)
            elif token == self.bos_token:
                token_ids.append(self.bos_token_id)
            elif token == self.eos_token:
                token_ids.append(self.eos_token_id)
            elif token == self.unk_token:
                token_ids.append(self.unk_token_id)
            else:
                # 학습 코드와 동일한 해시 방식: hash(token) % 65535 + 1
                token_id = hash(token) % 65535 + 1
                token_ids.append(token_id)
        
        return token_ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """ID를 토큰으로 변환 (근사치)"""
        tokens = []
        for token_id in ids:
            if token_id == self.pad_token_id:
                tokens.append(self.pad_token)
            elif token_id == self.bos_token_id:
                tokens.append(self.bos_token)
            elif token_id == self.eos_token_id:
                tokens.append(self.eos_token)
            elif token_id == self.unk_token_id:
                tokens.append(self.unk_token)
            else:
                # 해시는 비가역적이므로 토큰 ID를 문자열로 표현
                tokens.append(f"<token_{token_id}>")
        
        return tokens
    
    def encode(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
        """텍스트를 토큰 ID로 인코딩"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        input_ids = self.convert_tokens_to_ids(tokens)
        
        if return_tensors == "pt":
            import torch
            result = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long)
            }
            print(f"✅ 토크나이징 완료: {len(input_ids)}개 토큰 → shape {result['input_ids'].shape}")
            return result
        
        return input_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        if hasattr(token_ids, 'tolist'):  # torch.Tensor인 경우
            token_ids = token_ids.tolist()
        
        tokens = self.convert_ids_to_tokens(token_ids)
        
        if skip_special_tokens:
            # 특수 토큰 제거
            tokens = [
                token for token in tokens 
                if token not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
            ]
        
        # 토큰을 텍스트로 결합
        # 해시 기반 토큰은 원본 복원이 불가능하므로 토큰 ID 표시
        text_parts = []
        for token in tokens:
            if token.startswith("<token_"):
                # 해시 토큰은 [ID] 형태로 표시
                token_id = token.replace("<token_", "").replace(">", "")
                text_parts.append(f"[{token_id}]")
            else:
                text_parts.append(token)
        
        return " ".join(text_parts)
    
    def batch_encode_plus(self, texts: List[str], 
                         add_special_tokens: bool = True,
                         max_length: Optional[int] = None,
                         padding: bool = True,
                         truncation: bool = True,
                         return_tensors: Optional[str] = None) -> Dict[str, any]:
        """배치 인코딩"""
        all_input_ids = []
        all_attention_masks = []
        
        for text in texts:
            encoded = self.encode(
                text, 
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=False,  # 개별적으로는 패딩하지 않음
                truncation=truncation
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])
        
        # 배치 패딩
        if padding and max_length:
            for i in range(len(all_input_ids)):
                current_length = len(all_input_ids[i])
                if current_length < max_length:
                    pad_length = max_length - current_length
                    all_input_ids[i].extend([self.pad_token_id] * pad_length)
                    all_attention_masks[i].extend([0] * pad_length)
                else:
                    all_input_ids[i] = all_input_ids[i][:max_length]
                    all_attention_masks[i] = all_attention_masks[i][:max_length]
        
        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks
        }
        
        # 텐서 변환
        if return_tensors == "pt":
            import torch
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
            result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long)
        
        return result
    
    def save_pretrained(self, save_directory: str):
        """토크나이저 설정 저장"""
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "tokenizer_type": "TrainingCompatibleTokenizer"
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """사전 학습된 토크나이저 로드"""
        config_path = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"토크나이저 설정 로드: {config_path}")
        else:
            print("토크나이저 설정 파일이 없습니다. 기본 설정을 사용합니다.")
        
        return cls()
    
    def __len__(self):
        """어휘 크기 반환"""
        return self.vocab_size 