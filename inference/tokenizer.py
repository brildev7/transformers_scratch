"""
간단한 토크나이저 클래스
실제 프로덕션에서는 SentencePiece나 HuggingFace tokenizer를 사용하는 것을 권장합니다.
"""

import torch
import json
import os
from typing import List, Dict, Optional, Union


class SimpleTokenizer:
    """간단한 한국어 토크나이저"""
    
    def __init__(self, vocab_file: Optional[str] = None):
        """
        Args:
            vocab_file: 어휘 파일 경로 (선택적)
        """
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        # 특수 토큰 ID
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # 기본 어휘 초기화
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self.init_default_vocab()
    
    def init_default_vocab(self):
        """기본 어휘 초기화 (한국어 기본 문자 포함)"""
        # 특수 토큰
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # 한글 자모
        korean_chars = []
        # 한글 음절 (가-힣)
        for i in range(0xAC00, 0xD7A4):
            korean_chars.append(chr(i))
        
        # 영어 알파벳
        english_chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        english_chars += [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # 숫자
        numbers = [str(i) for i in range(10)]
        
        # 기본 구두점
        punctuation = [' ', '.', ',', '!', '?', ':', ';', '-', '_', '(', ')', '[', ']', '{', '}', '"', "'"]
        
        # 전체 어휘 구성
        all_tokens = special_tokens + korean_chars + english_chars + numbers + punctuation
        
        # 어휘 사전 생성
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"기본 어휘 초기화 완료: {self.vocab_size:,}개 토큰")
    
    def load_vocab(self, vocab_file: str):
        """어휘 파일에서 로드"""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                if vocab_file.endswith('.json'):
                    self.vocab = json.load(f)
                else:
                    # 텍스트 파일 형식 (한 줄에 하나씩)
                    tokens = [line.strip() for line in f if line.strip()]
                    self.vocab = {token: idx for idx, token in enumerate(tokens)}
            
            self.id_to_token = {idx: token for token, idx in self.vocab.items()}
            self.vocab_size = len(self.vocab)
            
            # 특수 토큰 ID 업데이트
            if self.pad_token in self.vocab:
                self.pad_token_id = self.vocab[self.pad_token]
            if self.unk_token in self.vocab:
                self.unk_token_id = self.vocab[self.unk_token]
            if self.bos_token in self.vocab:
                self.bos_token_id = self.vocab[self.bos_token]
            if self.eos_token in self.vocab:
                self.eos_token_id = self.vocab[self.eos_token]
            
            print(f"어휘 파일 로드 완료: {vocab_file}, {self.vocab_size:,}개 토큰")
            
        except Exception as e:
            print(f"어휘 파일 로드 실패: {e}")
            print("기본 어휘로 초기화합니다.")
            self.init_default_vocab()
    
    def save_vocab(self, vocab_file: str):
        """어휘를 파일로 저장"""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            if vocab_file.endswith('.json'):
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            else:
                for token in self.vocab.keys():
                    f.write(f"{token}\n")
        print(f"어휘 파일 저장 완료: {vocab_file}")
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할 (문자 단위)"""
        if not text:
            return []
        
        # 간단한 문자 단위 토크나이징
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        """텍스트를 토큰 ID로 인코딩"""
        tokens = self.tokenize(text)
        
        # 특수 토큰 추가
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # 토큰을 ID로 변환
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # 최대 길이 제한
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # 특수 토큰 스킵
                if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                    
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        return ''.join(tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True, 
                    max_length: Optional[int] = None, padding: bool = True) -> Dict[str, torch.Tensor]:
        """배치 텍스트 인코딩"""
        all_token_ids = []
        
        for text in texts:
            token_ids = self.encode(text, add_special_tokens, max_length)
            all_token_ids.append(token_ids)
        
        if padding and len(all_token_ids) > 1:
            # 패딩 처리
            max_len = max(len(ids) for ids in all_token_ids)
            if max_length:
                max_len = min(max_len, max_length)
            
            padded_ids = []
            attention_masks = []
            
            for token_ids in all_token_ids:
                # 패딩
                if len(token_ids) < max_len:
                    attention_mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
                    token_ids = token_ids + [self.pad_token_id] * (max_len - len(token_ids))
                else:
                    attention_mask = [1] * max_len
                    token_ids = token_ids[:max_len]
                
                padded_ids.append(token_ids)
                attention_masks.append(attention_mask)
            
            return {
                'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        else:
            # 패딩 없이 단일 시퀀스 또는 길이가 같은 시퀀스들
            return {
                'input_ids': torch.tensor(all_token_ids, dtype=torch.long)
            }
    
    def decode_batch(self, token_ids_batch: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """배치 토큰 ID를 텍스트로 디코딩"""
        texts = []
        for token_ids in token_ids_batch:
            text = self.decode(token_ids, skip_special_tokens)
            texts.append(text)
        return texts
    
    def __len__(self):
        """어휘 크기 반환"""
        return self.vocab_size
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """특수 토큰 정보 반환"""
        return {
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id
        } 