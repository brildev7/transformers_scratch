"""
개선된 학습 호환 토크나이저
해시 기반 토크나이징의 한계를 극복하기 위해 일반적인 한국어 단어들에 대한 고정 매핑 제공
"""

import json
import os
from typing import List, Dict, Optional, Union


class ImprovedTrainingCompatibleTokenizer:
    """학습 호환성을 유지하면서 디코딩 문제를 해결한 토크나이저"""
    
    def __init__(self, vocab_file: Optional[str] = None):
        """
        Args:
            vocab_file: 어휘 파일 경로 (선택적)
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
        
        # 일반적인 한국어 단어들에 대한 고정 매핑 (학습 시 사용된 해시값 기반)
        self._build_common_vocab()
        
        print(f"개선된 학습 호환 토크나이저 초기화 완료")
        print(f"  • 어휘 크기: {self.vocab_size:,}")
        print(f"  • 고정 매핑 단어 수: {len(self.word_to_id):,}")
        print(f"  • PAD: {self.pad_token_id}, BOS: {self.bos_token_id}, EOS: {self.eos_token_id}, UNK: {self.unk_token_id}")
        
    def _build_common_vocab(self):
        """자주 사용되는 한국어 단어들에 대한 고정 매핑 구축"""
        common_words = [
            # 인사
            "안녕하세요", "안녕", "반갑습니다", "감사합니다", "죄송합니다",
            # 기본 명사
            "사람", "집", "학교", "회사", "물", "음식", "시간", "날씨", "돈",
            "한국", "서울", "부산", "대구", "인천", "광주", "대전", "울산",
            # 기본 동사
            "가다", "오다", "보다", "듣다", "말하다", "먹다", "마시다", "자다", "일어나다",
            "하다", "되다", "있다", "없다", "좋다", "나쁘다", "크다", "작다",
            # 기본 형용사/부사
            "매우", "정말", "아주", "조금", "많이", "빨리", "천천히", "잘", "못",
            # 대명사
            "나", "너", "우리", "그들", "이것", "그것", "저것", "여기", "거기", "저기",
            # 수사
            "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열",
            # 기타 자주 사용되는 단어들
            "네", "아니오", "예", "안", "못", "또", "그리고", "하지만", "그러나",
            "어떤", "무엇", "누구", "언제", "어디", "왜", "어떻게",
            # 기술/AI 관련 용어
            "모델", "데이터", "학습", "추론", "AI", "인공지능", "컴퓨터", "프로그램",
            "텍스트", "문장", "단어", "토큰", "생성", "예측", "결과", "성능",
            # 일반적인 동사 활용형
            "합니다", "입니다", "갑니다", "옵니다", "봅니다", "듣습니다", "말합니다",
            "먹습니다", "마십니다", "잡니다", "일어납니다", "됩니다", "좋습니다",
            # 조사
            "은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로", "와", "과",
            "의", "도", "만", "부터", "까지", "처럼", "같이", "보다", "마다"
        ]
        
        # 단어 → ID 매핑 구축 (학습 시와 동일한 해시 방식)
        self.word_to_id = {}
        self.id_to_word = {}
        
        for word in common_words:
            word_id = hash(word) % 65535 + 1  # 학습 코드와 동일한 해시 방식
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
        
        # 자주 생성되는 토큰 ID들에 대한 추가 매핑 (의미 추정)
        frequent_tokens = {
            17356: "그런데",    # 연결어 추정
            21857: "정말",      # 부사 추정  
            50820: "그리고",    # 연결어 추정
            20390: "좋다",      # 형용사 추정
            3031: "많이",       # 부사 추정
            61151: "하지만",    # 연결어 추정
            1400: "또",         # 연결어 추정
            11459: "아주",      # 부사 추정
            19034: "그래서",    # 연결어 추정
            30346: "정말로",    # 부사 추정
            34772: "매우",      # 부사 추정
            18171: "그러나",    # 연결어 추정
            59820: "되다",      # 동사 추정
            41887: "있다",      # 동사 추정
            14944: "잘",        # 부사 추정
            8749: "더",         # 부사 추정
            39047: "크다",      # 형용사 추정
            33897: "또한",      # 연결어 추정
            8306: "그것",       # 대명사 추정
            9203: "이것",       # 대명사 추정
            56615: "따라서",    # 연결어 추정
            35299: "너무",      # 부사 추정
        }
        
        # 추정 매핑 추가
        for token_id, estimated_word in frequent_tokens.items():
            if token_id not in self.id_to_word:  # 기존 매핑과 충돌하지 않는 경우만
                self.id_to_word[token_id] = f"{estimated_word}*"  # *로 추정임을 표시
                
        print(f"  • 추정 매핑 추가: {len(frequent_tokens)}개")
    
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
        """ID를 토큰으로 변환 (개선된 역매핑)"""
        tokens = []
        for token_id in ids:
            if token_id == self.pad_token_id:
                tokens.append("")  # PAD 토큰은 빈 문자열로 처리
            elif token_id == self.bos_token_id:
                tokens.append("")  # BOS 토큰도 출력에서 숨김
            elif token_id == self.eos_token_id:
                tokens.append("")  # EOS 토큰도 출력에서 숨김
            elif token_id == self.unk_token_id:
                tokens.append("[UNK]")
            elif token_id in self.id_to_word:
                # 고정 매핑에서 찾은 경우 - 실제 한국어 단어
                tokens.append(self.id_to_word[token_id])
            else:
                # 매핑에 없는 토큰은 간결하게 표시
                tokens.append(f"[{token_id}]")
        
        return tokens
        
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
        """토큰 ID를 텍스트로 디코딩 (개선된 버전)"""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        tokens = self.convert_ids_to_tokens(token_ids)
        
        # 빈 문자열과 불필요한 토큰 제거
        filtered_tokens = []
        for token in tokens:
            if token and token.strip():  # 빈 문자열이 아닌 경우만
                filtered_tokens.append(token)
        
        # 토큰들을 공백으로 연결
        return " ".join(filtered_tokens)
    def get_vocab_coverage(self, text: str) -> Dict[str, float]:
        """주어진 텍스트에 대한 어휘 커버리지 확인"""
        tokens = self.tokenize(text)
        total_tokens = len(tokens)
        covered_tokens = sum(1 for token in tokens if token in self.word_to_id)
        
        coverage = covered_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return {
            "total_tokens": total_tokens,
            "covered_tokens": covered_tokens,
            "coverage_ratio": coverage,
            "uncovered_tokens": [token for token in tokens if token not in self.word_to_id]
        }
    
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
                return_tensors=None  # 개별적으로는 텐서로 변환하지 않음
            )
            all_input_ids.append(encoded)
            all_attention_masks.append([1] * len(encoded))
        
        # 패딩 처리
        if padding and max_length:
            for i in range(len(all_input_ids)):
                current_length = len(all_input_ids[i])
                if current_length < max_length:
                    padding_length = max_length - current_length
                    all_input_ids[i].extend([self.pad_token_id] * padding_length)
                    all_attention_masks[i].extend([0] * padding_length)
                elif truncation and current_length > max_length:
                    all_input_ids[i] = all_input_ids[i][:max_length]
                    all_attention_masks[i] = all_attention_masks[i][:max_length]
        
        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks
        }
        
        if return_tensors == "pt":
            import torch
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
            result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long)
        
        return result


# 기존 TrainingCompatibleTokenizer를 개선된 버전으로 대체하는 팩토리 함수
def create_improved_tokenizer(vocab_file: Optional[str] = None):
    """개선된 토크나이저 생성"""
    return ImprovedTrainingCompatibleTokenizer(vocab_file)


if __name__ == "__main__":
    # 테스트
    tokenizer = ImprovedTrainingCompatibleTokenizer()
    
    test_texts = [
        "안녕하세요 한국어 테스트입니다",
        "모델 학습이 잘 되고 있습니다",
        "AI 기술이 발전하고 있습니다"
    ]
    
    print("\n" + "=" * 60)
    print("🧪 개선된 토크나이저 테스트")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\n원본: '{text}'")
        
        # 어휘 커버리지 확인
        coverage = tokenizer.get_vocab_coverage(text)
        print(f"커버리지: {coverage['coverage_ratio']:.1%} ({coverage['covered_tokens']}/{coverage['total_tokens']})")
        if coverage['uncovered_tokens']:
            print(f"매핑되지 않은 토큰: {coverage['uncovered_tokens']}")
        
        # 인코딩/디코딩 테스트
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        decoded = tokenizer.decode(token_ids)
        
        print(f"토큰: {tokens}")
        print(f"토큰 ID: {token_ids}")
        print(f"디코딩: '{decoded}'") 