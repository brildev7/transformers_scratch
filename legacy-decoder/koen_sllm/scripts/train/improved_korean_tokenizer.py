#!/usr/bin/env python3
"""
개선된 한국어 토크나이저
Improved Korean Tokenizer

공백 기준 토큰화와 학습 데이터 기반 어휘 구축을 지원합니다.
- 단어 단위 맥락 보존
- 학습 데이터 기반 어휘 사전 구축
- Subword 토큰화 지원
- 한국어 특성 고려
"""

import re
import json
import os
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from collections import Counter
import pickle


class ImprovedKoreanTokenizer:
    """개선된 한국어 토크나이저"""
    
    def __init__(self, vocab_file: Optional[str] = None, vocab_size: int = 32000):
        """
        Args:
            vocab_file: 사전 구축된 어휘 파일 경로
            vocab_size: 어휘 크기
        """
        # 특수 토큰
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        
        # 특수 토큰 ID
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        # 어휘 크기
        self.vocab_size = vocab_size
        
        # 어휘 매핑
        self.word_to_id = {}
        self.id_to_word = {}
        
        # 한국어 전처리 패턴
        self._setup_korean_patterns()
        
        # 어휘 로드 또는 초기화
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocabulary(vocab_file)
        else:
            self._initialize_basic_vocab()
            
        print(f"개선된 한국어 토크나이저 초기화 완료")
        print(f"  • 어휘 크기: {len(self.word_to_id):,} / {self.vocab_size:,}")
        print(f"  • 토큰화 방식: 공백 기준 + Subword")
        
    def _setup_korean_patterns(self):
        """한국어 전처리 패턴 설정"""
        
        # 한국어 문자 패턴
        self.korean_pattern = re.compile(r'[가-힣]+')
        
        # 영어 패턴  
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        
        # 숫자 패턴
        self.number_pattern = re.compile(r'\d+')
        
        # 구두점 패턴
        self.punctuation_pattern = re.compile(r'[.!?,:;"\'\(\)\[\]\{\}~\-_+=<>/@#$%^&*`|\\]')
        
        # 자주 사용되는 한국어 조사/어미 (분리하지 않고 단어와 함께 유지)
        self.common_endings = {
            '이다', '이야', '에요', '어요', '습니다', '입니다', '했다', '한다', '된다',
            '이는', '에는', '에서', '으로', '로써', '에게', '한테', '에게서'
        }
        
    def _initialize_basic_vocab(self):
        """기본 어휘 초기화"""
        
        # 특수 토큰부터 할당
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            
        # 기본 한국어 어휘 (자주 사용되는 단어들)
        basic_korean_words = [
            # 인사/기본
            "안녕하세요", "안녕히", "가세요", "반갑습니다", "감사합니다", "죄송합니다",
            "괜찮습니다", "네", "아니오", "예", "아니", "맞습니다", "틀렸습니다",
            
            # 대명사
            "나", "너", "우리", "그들", "이것", "그것", "저것", "여기", "거기", "저기",
            
            # 동사 (기본형)
            "하다", "되다", "있다", "없다", "가다", "오다", "보다", "듣다", "말하다",
            "읽다", "쓰다", "먹다", "마시다", "자다", "일어나다", "앉다", "서다",
            
            # 형용사 (기본형)
            "좋다", "나쁘다", "크다", "작다", "많다", "적다", "길다", "짧다",
            "높다", "낮다", "넓다", "좁다", "기쁘다", "슬프다", "화나다",
            
            # 명사 - 기본
            "사람", "시간", "날", "년", "월", "일", "시", "분", "초",
            "집", "학교", "회사", "병원", "가게", "식당", "카페",
            
            # AI/기술 용어
            "모델", "데이터", "학습", "추론", "텍스트", "토큰", "문장", "단어",
            "AI", "인공지능", "컴퓨터", "프로그램", "소프트웨어"
        ]
        
        # 기본 어휘 할당
        current_id = len(special_tokens)
        for word in basic_korean_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
                
        print(f"  • 기본 어휘 할당: {len(basic_korean_words)}개")
        
    def build_vocabulary_from_data(self, data_paths: List[str], save_path: Optional[str] = None):
        """학습 데이터로부터 어휘 구축"""
        
        print("📚 학습 데이터로부터 어휘 구축 중...")
        
        # 모든 텍스트 수집
        all_texts = []
        for data_path in data_paths:
            if os.path.exists(data_path):
                if data_path.endswith('.jsonl'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                all_texts.append(data.get('text', ''))
                elif data_path.endswith('.txt'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        all_texts.extend(f.readlines())
                        
        print(f"  • 총 {len(all_texts)}개 텍스트 수집")
        
        # 토큰 빈도 계산
        token_counter = Counter()
        for text in all_texts:
            tokens = self._tokenize_raw(text)
            token_counter.update(tokens)
            
        print(f"  • 유니크 토큰 수: {len(token_counter):,}")
        
        # 빈도 순으로 어휘 구축
        current_id = len(self.word_to_id)
        vocab_limit = self.vocab_size - 100  # 여유 공간
        
        for token, freq in token_counter.most_common():
            if current_id >= vocab_limit:
                break
            if token not in self.word_to_id and len(token.strip()) > 0:
                self.word_to_id[token] = current_id
                self.id_to_word[current_id] = token
                current_id += 1
                
        print(f"  • 최종 어휘 크기: {len(self.word_to_id):,}")
        
        # 어휘 저장
        if save_path:
            self.save_vocabulary(save_path)
            
    def _tokenize_raw(self, text: str) -> List[str]:
        """원시 텍스트를 토큰으로 분할 (어휘 구축용)"""
        
        # 기본 전처리
        text = text.strip()
        if not text:
            return []
            
        # 공백 기준 분할
        words = text.split()
        
        tokens = []
        for word in words:
            # 구두점이 붙어있는 경우 분리
            word = word.strip()
            if not word:
                continue
                
            # 구두점 분리
            if self.punctuation_pattern.search(word):
                # 간단한 구두점 분리
                parts = re.split(r'([.!?,:;"\'\(\)\[\]\{\}])', word)
                for part in parts:
                    if part.strip():
                        tokens.append(part.strip())
            else:
                tokens.append(word)
                
        return tokens
        
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할 (추론용)"""
        
        if not text.strip():
            return []
            
        # 공백 기준 토큰화
        tokens = self._tokenize_raw(text)
        
        # OOV 처리: 서브워드 분할
        final_tokens = []
        for token in tokens:
            if token in self.word_to_id:
                final_tokens.append(token)
            else:
                # OOV 토큰을 서브워드로 분할
                subwords = self._split_to_subwords(token)
                final_tokens.extend(subwords)
                
        return final_tokens
        
    def _split_to_subwords(self, word: str) -> List[str]:
        """OOV 단어를 서브워드로 분할"""
        
        # 길이가 짧으면 그대로 반환
        if len(word) <= 2:
            return [word]
            
        # 한국어인 경우 음절 단위로 분할
        if self.korean_pattern.match(word):
            # 2-3음절씩 분할
            subwords = []
            i = 0
            while i < len(word):
                if i + 3 <= len(word):
                    subword = word[i:i+3]
                    if subword in self.word_to_id:
                        subwords.append(subword)
                        i += 3
                        continue
                        
                if i + 2 <= len(word):
                    subword = word[i:i+2]
                    if subword in self.word_to_id:
                        subwords.append(subword)
                        i += 2
                        continue
                        
                # 단일 음절
                subwords.append(word[i])
                i += 1
                
            return subwords
        else:
            # 영어/기타의 경우 prefix 방식
            subwords = []
            remaining = word
            while remaining:
                found = False
                for length in range(min(len(remaining), 6), 0, -1):
                    prefix = remaining[:length]
                    if prefix in self.word_to_id:
                        subwords.append(prefix)
                        remaining = remaining[length:]
                        found = True
                        break
                        
                if not found:
                    # 단일 문자로 분할
                    subwords.append(remaining[0])
                    remaining = remaining[1:]
                    
            return subwords
            
    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None, return_tensors: Optional[str] = None, **kwargs) -> Dict:
        """텍스트를 토큰 ID로 인코딩"""
        
        tokens = self.tokenize(text)
        
        # 특수 토큰 추가
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
            
        # 토큰 → ID 변환
        input_ids = []
        for token in tokens:
            if token in self.word_to_id:
                input_ids.append(self.word_to_id[token])
            else:
                # 최후의 수단: 해시 기반 ID
                token_id = hash(token) % (self.vocab_size - 1000) + 1000
                input_ids.append(token_id)
                
        # 길이 제한
        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            
        # attention_mask 생성
        attention_mask = [1] * len(input_ids)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tokens": tokens
        }
        
        # PyTorch 텐서 변환
        if return_tensors == "pt":
            try:
                import torch
                result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long).unsqueeze(0)
                result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long).unsqueeze(0)
            except ImportError:
                pass
                
        return result
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(f"[{token_id}]")
                    
        # 공백으로 연결 (자연스러운 한국어 문장 생성)
        return " ".join(tokens)
        
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return len(self.word_to_id)
        
    def save_vocabulary(self, save_path: str):
        """어휘 저장"""
        vocab_data = {
            "word_to_id": self.word_to_id,
            "id_to_word": {str(k): v for k, v in self.id_to_word.items()},
            "vocab_size": self.vocab_size,
            "tokenizer_type": "improved_korean"
        }
        
        # JSON 저장
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
        # Pickle 저장 (빠른 로딩용)
        pickle_path = save_path.replace('.json', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(vocab_data, f)
            
        print(f"어휘 저장 완료: {save_path}")
        
    def _load_vocabulary(self, vocab_file: str):
        """저장된 어휘 로드"""
        
        try:
            # Pickle 파일 우선 시도
            pickle_file = vocab_file.replace('.json', '.pkl')
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    vocab_data = pickle.load(f)
            else:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                    
            self.word_to_id = vocab_data["word_to_id"]
            self.id_to_word = {int(k): v for k, v in vocab_data["id_to_word"].items()}
            self.vocab_size = vocab_data.get("vocab_size", 32000)
            
            print(f"어휘 로드 완료: {len(self.word_to_id):,}개")
            
        except Exception as e:
            print(f"⚠️ 어휘 로드 실패: {e}")
            self._initialize_basic_vocab()


# 테스트 및 검증
if __name__ == "__main__":
    print("=" * 60)
    print("🇰🇷 개선된 한국어 토크나이저 테스트")
    print("=" * 60)
    
    # 개선된 한국어 토크나이저 테스트
    tokenizer = ImprovedKoreanTokenizer()
    
    test_sentences = [
        "안녕하세요, 저는 한국어 언어모델입니다.",
        "공백 기준으로 토큰화하여 맥락을 보존합니다.",
        "AI 모델이 한국어를 더 잘 이해할 수 있어요!",
        "데이터 과학과 자연어 처리는 흥미로운 분야입니다."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n테스트 {i}: {sentence}")
        
        # 토크나이징
        tokens = tokenizer.tokenize(sentence)
        print(f"  토큰들: {tokens}")
        print(f"  토큰 수: {len(tokens)}")
        
        # 인코딩
        encoded = tokenizer.encode(sentence)
        print(f"  토큰 ID들: {encoded['input_ids'][:10]}...")
        
        # 디코딩
        decoded = tokenizer.decode(encoded['input_ids'])
        print(f"  디코딩: {decoded}")
        
    print(f"\n📊 최종 통계:")
    print(f"  • 개선된 한국어 토크나이저 어휘: {tokenizer.get_vocab_size():,}")
    print(f"  • 맥락 보존형 토크나이징으로 한국어 모델 성능 향상 기대!") 