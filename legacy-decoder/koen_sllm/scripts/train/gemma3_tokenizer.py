#!/usr/bin/env python3
"""
Gemma3 토크나이저 모듈

Google의 Gemma3 모델의 토크나이저를 래핑하여 
한국어 sLLM 학습에 사용할 수 있도록 구현한 모듈입니다.

Features:
- Google 검증된 고성능 토크나이저
- 다국어 지원 (한국어 포함)
- 일관된 API 인터페이스
- 서브워드 토크나이징

Author: AI Assistant
Date: 2025-07-29
Version: 1.0.0
"""

import logging
from typing import List, Dict, Optional, Union
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Gemma3TokenizerWrapper:
    """
    Gemma3 토크나이저 래퍼 클래스
    
    Google의 Gemma3 모델에서 사용하는 토크나이저를 
    한국어 sLLM 학습에 맞게 래핑한 클래스입니다.
    
    Features:
        - 다국어 지원 (한국어 포함)
        - 서브워드 토크나이징
        - 일관된 API 인터페이스
        - PyTorch 텐서 지원
    """
    
    def __init__(self, model_name: str = "google/gemma-2-2b"):
        """
        Gemma3 토크나이저 초기화
        
        Args:
            model_name (str): 사용할 Gemma 모델 이름
                - google/gemma-2-2b (기본값, 추천)
                - google/gemma-2-9b
                - google/gemma-2-27b
        
        Raises:
            ImportError: transformers 라이브러리가 없는 경우
            RuntimeError: 모델 로드에 실패한 경우
        """
        self.model_name = model_name
        
        try:
            from transformers import AutoTokenizer
            
            logger.info(f"🤖 Gemma3 토크나이저 로드 중: {model_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True  # 빠른 토크나이저 사용
            )
            
            # 어휘 크기 확인
            self.vocab_size = self.tokenizer.vocab_size
            
            # 특수 토큰 설정
            self._setup_special_tokens()
            
            logger.info(f"✅ Gemma3 토크나이저 로드 완료")
            logger.info(f"  • 모델: {model_name}")
            logger.info(f"  • 어휘 크기: {self.vocab_size:,}")
            logger.info(f"  • PAD 토큰: {self.tokenizer.pad_token}")
            logger.info(f"  • BOS 토큰: {self.tokenizer.bos_token}")
            logger.info(f"  • EOS 토큰: {self.tokenizer.eos_token}")
            
        except ImportError as e:
            error_msg = (
                "transformers 라이브러리가 필요합니다.\n"
                "설치 방법: pip install transformers>=4.35.0"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Gemma3 토크나이저 로드 실패: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _setup_special_tokens(self):
        """특수 토큰 설정"""
        # PAD 토큰이 없으면 EOS 토큰을 PAD로 사용
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("PAD 토큰을 EOS 토큰으로 설정")
    
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트를 토큰으로 분할
        
        Args:
            text (str): 토크나이징할 텍스트
            
        Returns:
            List[str]: 토큰 리스트
        """
        try:
            return self.tokenizer.tokenize(text)
        except Exception as e:
            logger.warning(f"토크나이징 실패: {e}")
            # 기본 공백 분할로 fallback
            return text.split()
    
    def encode(self, 
               text: str, 
               add_special_tokens: bool = True, 
               max_length: Optional[int] = None, 
               return_tensors: Optional[str] = None, 
               **kwargs) -> Dict:
        """
        텍스트를 토큰 ID로 인코딩
        
        Args:
            text (str): 인코딩할 텍스트
            add_special_tokens (bool): 특수 토큰 추가 여부
            max_length (int, optional): 최대 길이 (자동 자르기)
            return_tensors (str, optional): 반환 텐서 타입 ("pt", "tf", None)
            **kwargs: 추가 토크나이저 옵션
            
        Returns:
            Dict: 인코딩 결과
                - input_ids: 토큰 ID 리스트
                - attention_mask: 어텐션 마스크
                - tokens: 토큰 리스트 (디버깅용)
        """
        try:
            # 토크나이저 호출
            encoding = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=max_length is not None,
                padding=False,  # 수동으로 패딩 처리
                return_tensors=return_tensors,
                **kwargs
            )
            
            # 토큰 정보 추가 (디버깅용)
            if return_tensors is None:
                tokens = self.tokenizer.tokenize(text)
                encoding['tokens'] = tokens
            
            return encoding
            
        except Exception as e:
            logger.error(f"인코딩 실패: {e}")
            # Fallback: 기본 처리
            tokens = text.split()
            input_ids = [1] + [hash(token) % (self.vocab_size - 4) + 4 for token in tokens] + [2]
            attention_mask = [1] * len(input_ids)
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'tokens': tokens
            }
            
            if return_tensors == "pt":
                result['input_ids'] = torch.tensor([result['input_ids']], dtype=torch.long)
                result['attention_mask'] = torch.tensor([result['attention_mask']], dtype=torch.long)
                
            return result
    
    def decode(self, 
               token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True) -> str:
        """
        토큰 ID를 텍스트로 디코딩
        
        Args:
            token_ids: 디코딩할 토큰 ID 리스트 또는 텐서
            skip_special_tokens (bool): 특수 토큰 제거 여부
            
        Returns:
            str: 디코딩된 텍스트
        """
        try:
            # 텐서를 리스트로 변환
            if torch.is_tensor(token_ids):
                if token_ids.dim() > 1:
                    token_ids = token_ids.squeeze()
                token_ids = token_ids.tolist()
            
            return self.tokenizer.decode(
                token_ids, 
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True
            )
            
        except Exception as e:
            logger.warning(f"디코딩 실패: {e}")
            # Fallback: 토큰 ID를 문자열로 변환
            return " ".join([f"[{tid}]" for tid in token_ids])
    
    def get_vocab_size(self) -> int:
        """
        어휘 크기 반환
        
        Returns:
            int: 어휘 크기
        """
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, str]:
        """
        특수 토큰 정보 반환
        
        Returns:
            Dict[str, str]: 특수 토큰 딕셔너리
        """
        return {
            'pad_token': self.tokenizer.pad_token,
            'bos_token': self.tokenizer.bos_token,
            'eos_token': self.tokenizer.eos_token,
            'unk_token': self.tokenizer.unk_token,
        }
    
    def save_pretrained(self, save_directory: str):
        """
        토크나이저 저장
        
        Args:
            save_directory (str): 저장할 디렉토리 경로
        """
        try:
            self.tokenizer.save_pretrained(save_directory)
            logger.info(f"토크나이저 저장 완료: {save_directory}")
        except Exception as e:
            logger.error(f"토크나이저 저장 실패: {e}")
            raise


# 테스트 및 검증
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Gemma3 토크나이저 테스트")
    print("=" * 60)
    
    try:
        # Gemma3 토크나이저 초기화
        tokenizer = Gemma3TokenizerWrapper()
        
        # 테스트 문장들
        test_sentences = [
            "안녕하세요, 저는 한국어 AI 모델입니다.",
            "Gemma3 tokenizer는 다국어를 지원합니다.",
            "자연어 처리와 머신러닝은 매우 흥미로운 분야예요!",
            "Hello, world! This is a test sentence."
        ]
        
        print(f"\n📊 토크나이저 정보:")
        print(f"  • 모델: {tokenizer.model_name}")
        print(f"  • 어휘 크기: {tokenizer.get_vocab_size():,}")
        
        special_tokens = tokenizer.get_special_tokens()
        print(f"  • 특수 토큰: {special_tokens}")
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n🧪 테스트 {i}: {sentence}")
            
            # 토크나이징
            tokens = tokenizer.tokenize(sentence)
            print(f"  📝 토큰들: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"  📊 토큰 수: {len(tokens)}")
            
            # 인코딩
            encoded = tokenizer.encode(sentence, return_tensors="pt")
            input_ids = encoded['input_ids'].squeeze().tolist()
            print(f"  🔢 토큰 ID들: {input_ids[:10]}{'...' if len(input_ids) > 10 else ''}")
            
            # 디코딩
            decoded = tokenizer.decode(input_ids)
            print(f"  🔄 디코딩: {decoded}")
            
            # 일치성 확인
            is_match = sentence.strip() == decoded.strip()
            print(f"  ✅ 일치성: {'✓' if is_match else '✗'}")
            
        print(f"\n🎉 Gemma3 토크나이저 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("🔧 해결방법:")
        print("  1. pip install transformers>=4.35.0")
        print("  2. 인터넷 연결 확인")
        print("  3. CUDA 메모리 확인") 