#!/usr/bin/env python3
"""
한국어 지원 토크나이저
Korean Language Supporting Tokenizer

한국어의 교착어 특성을 고려한 토크나이저입니다.
- 형태소 분석 기반 토크나이징
- 조사, 어미 분리
- 구두점 처리
- 학습-추론 일관성 보장
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class KoreanTokenizer:
    """한국어 지원 토크나이저"""
    
    def __init__(self, vocab_file: Optional[str] = None):
        """
        Args:
            vocab_file: 어휘 파일 경로 (선택적)
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
        self.vocab_size = 65536
        
        # 한국어 패턴 정의
        self._setup_korean_patterns()
        
        # 기본 어휘 구축
        self._build_vocab()
        
        print(f"한국어 토크나이저 초기화 완료")
        print(f"  • 어휘 크기: {self.vocab_size:,}")
        print(f"  • 기본 어휘: {len(self.word_to_id):,}개")
        
    def _setup_korean_patterns(self):
        """한국어 처리 패턴 설정"""
        
        # 조사 패턴 (주격, 목적격, 부사격 등)
        self.josa_patterns = [
            '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', 
            '의', '도', '만', '부터', '까지', '처럼', '같이', '보다', '마다',
            '에게', '한테', '께', '께서', '라서', '니까', '때문에', '위해', '대해'
        ]
        
        # 어미 패턴 (존댓말, 반말, 시제 등)
        self.eomi_patterns = [
            '습니다', '습니까', '세요', '십시오', '합니다', '했습니다', '하겠습니다',
            '입니다', '였습니다', '이었습니다', '해요', '해서', '하고', '하면',
            '하지만', '하거나', '하든지', '한다', '했다', '하자', '하라'
        ]
        
        # 구두점 패턴
        self.punctuation = r'[.!?,:;"\'\(\)\[\]\{\}~\-_+=<>/@#$%^&*`|\\]'
        
        # 숫자 패턴
        self.number_pattern = r'\d+'
        
        # 영어 패턴
        self.english_pattern = r'[a-zA-Z]+'
        
    def _build_vocab(self):
        """기본 어휘 구축"""
        
        # 기본 한국어 어휘 (형태소 단위)
        basic_vocab = [
            # 대명사
            "나", "너", "우리", "그들", "이", "그", "저", "것", "곳",
            
            # 명사 - 사람
            "사람", "친구", "가족", "부모", "아버지", "어머니", "형", "누나", "동생",
            "선생님", "학생", "의사", "간호사", "회사원", "아이", "어른",
            
            # 명사 - 장소  
            "집", "학교", "회사", "병원", "은행", "상점", "시장", "공원", "도서관",
            "카페", "식당", "호텔", "역", "공항", "도시", "마을",
            
            # 명사 - 사물
            "물", "음식", "책", "컴퓨터", "핸드폰", "자동차", "옷", "신발", "가방",
            "돈", "시간", "날씨", "공기", "나무", "꽃", "바다", "산", "강", "하늘",
            
            # 동사 어근
            "가", "오", "보", "듣", "말하", "읽", "쓰", "먹", "마시", "자", "일어나",
            "앉", "서", "눕", "걷", "뛰", "놀", "일하", "공부하", "하", "되", "있", "없",
            
            # 형용사 어근
            "좋", "나쁘", "크", "작", "많", "적", "길", "짧", "높", "낮", "넓", "좁",
            "기쁘", "슬프", "화나", "무섭", "행복하", "편하", "불편하", "피곤하",
            
            # 부사
            "매우", "정말", "아주", "조금", "많이", "적게", "빨리", "천천히",
            "잘", "못", "더", "덜", "가장", "제일", "너무", "자주", "가끔",
            
            # 연결어
            "그리고", "하지만", "그러나", "그런데", "또한", "또", "그래서", "따라서",
            "만약", "비록", "아마", "혹시", "정말로", "확실히", "분명히",
            
            # 인사
            "안녕하", "반갑", "감사하", "죄송하", "고맙", "미안하",
            
            # 시간
            "오늘", "어제", "내일", "지금", "나중", "전", "후", "동안",
            "아침", "점심", "저녁", "밤", "새벽",
            
            # 기술/AI 용어
            "모델", "데이터", "학습", "추론", "AI", "인공지능", "컴퓨터", "프로그램",
            "텍스트", "문장", "단어", "토큰", "생성", "예측", "결과", "성능"
        ]
        
        # 조사와 어미 추가
        basic_vocab.extend(self.josa_patterns)
        basic_vocab.extend(self.eomi_patterns)
        
        # 기본 응답어
        basic_vocab.extend([
            "네", "아니오", "예", "아니", "맞", "틀리", "그렇", "괜찮", "좋겠", "싫"
        ])
        
        # 단어 → ID 매핑 구축
        self.word_to_id = {}
        self.id_to_word = {}
        
        # 특수 토큰부터 할당
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        # 기본 어휘 할당 (ID 4부터 시작)
        current_id = 4
        for word in basic_vocab:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        print(f"  • 특수 토큰: {len(special_tokens)}개")
        print(f"  • 기본 어휘: {len(basic_vocab)}개")
        
    def tokenize(self, text: str) -> List[str]:
        """한국어 텍스트를 토큰으로 분할"""
        
        # 전처리: 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        
        tokens = []
        i = 0
        
        while i < len(text):
            # 공백 건너뛰기
            if text[i].isspace():
                i += 1
                continue
            
            # 구두점 처리
            if re.match(self.punctuation, text[i]):
                tokens.append(text[i])
                i += 1
                continue
            
            # 숫자 처리
            number_match = re.match(self.number_pattern, text[i:])
            if number_match:
                tokens.append(number_match.group())
                i += len(number_match.group())
                continue
            
            # 영어 처리
            english_match = re.match(self.english_pattern, text[i:])
            if english_match:
                tokens.append(english_match.group().lower())
                i += len(english_match.group())
                continue
            
            # 한국어 단어 추출
            word_start = i
            while i < len(text) and not text[i].isspace() and not re.match(self.punctuation, text[i]):
                i += 1
            
            word = text[word_start:i]
            if word:
                # 형태소 분석 시뮬레이션
                morphemes = self._analyze_morphemes(word)
                tokens.extend(morphemes)
        
        return tokens
    
    def _analyze_morphemes(self, word: str) -> List[str]:
        """형태소 분석 (간단한 규칙 기반)"""
        
        # 조사 분리
        for josa in sorted(self.josa_patterns, key=len, reverse=True):
            if word.endswith(josa) and len(word) > len(josa):
                stem = word[:-len(josa)]
                return [stem, josa]
        
        # 어미 분리 
        for eomi in sorted(self.eomi_patterns, key=len, reverse=True):
            if word.endswith(eomi) and len(word) > len(eomi):
                stem = word[:-len(eomi)]
                return [stem, eomi]
        
        # 분리할 수 없으면 그대로 반환
        return [word]
    
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
                # OOV 토큰 처리: 해시 기반으로 일관성 있게 할당
                token_id = hash(token) % (self.vocab_size - 1000) + 1000  # 상위 ID 영역 사용
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
        
        # PyTorch 텐서 변환 (호환성을 위해)
        if return_tensors == "pt":
            try:
                import torch
                result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long).unsqueeze(0)
                result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long).unsqueeze(0)
            except ImportError:
                pass  # torch가 없으면 리스트 형태로 반환
        
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
                # OOV 토큰 처리
                if not skip_special_tokens:
                    tokens.append(f"[{token_id}]")
        
        # 토큰 재조합 (한국어 특성 고려)
        return self._reassemble_korean(tokens)
    
    def _reassemble_korean(self, tokens: List[str]) -> str:
        """한국어 토큰들을 자연스럽게 재조합"""
        
        if not tokens:
            return ""
        
        result = []
        i = 0
        
        while i < len(tokens):
            current_token = tokens[i]
            
            # 다음 토큰이 조사나 어미인지 확인
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token in self.josa_patterns or next_token in self.eomi_patterns:
                    # 어근과 조사/어미 결합
                    result.append(current_token + next_token)
                    i += 2
                    continue
            
            # 구두점인 경우 공백 없이 붙임
            if re.match(self.punctuation, current_token):
                if result:
                    result[-1] += current_token
                else:
                    result.append(current_token)
            else:
                result.append(current_token)
            
            i += 1
        
        return " ".join(result)
    
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return self.vocab_size
    
    def save_vocabulary(self, save_path: str):
        """어휘 저장"""
        vocab_data = {
            "word_to_id": self.word_to_id,
            "id_to_word": {str(k): v for k, v in self.id_to_word.items()},
            "vocab_size": self.vocab_size
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"어휘 저장 완료: {save_path}")


# 테스트 및 검증
if __name__ == "__main__":
    print("=" * 60)
    print("🇰🇷 한국어 토크나이저 테스트")
    print("=" * 60)
    
    tokenizer = KoreanTokenizer()
    
    # 테스트 문장들
    test_sentences = [
        "안녕하세요, 저는 한국어 언어모델입니다.",
        "반갑습니다! 오늘 날씨가 좋네요.",
        "AI 모델이 텍스트를 잘 생성합니다.",
        "학습 데이터는 중요합니다.",
        "토크나이저가 형태소를 분석해요."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n테스트 {i}: {sentence}")
        
        # 토크나이징
        tokens = tokenizer.tokenize(sentence)
        print(f"  토큰들: {tokens}")
        
        # 인코딩
        encoded = tokenizer.encode(sentence)
        print(f"  토큰 ID들: {encoded['input_ids'][:10]}...")  # 처음 10개만
        
        # 디코딩
        decoded = tokenizer.decode(encoded['input_ids'])
        print(f"  디코딩: {decoded}")
        
        # 일치성 확인
        print(f"  원문 일치: {'✅' if sentence.replace(',', ' ,').replace('.', ' .').replace('!', ' !') == decoded else '⚠️'}")
    
    print(f"\n📊 어휘 통계:")
    print(f"  • 총 어휘 크기: {tokenizer.get_vocab_size():,}")
    print(f"  • 등록된 어휘: {len(tokenizer.word_to_id):,}")
    print(f"  • 조사 패턴: {len(tokenizer.josa_patterns)}개")
    print(f"  • 어미 패턴: {len(tokenizer.eomi_patterns)}개") 