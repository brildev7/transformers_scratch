"""
Korean-English Tokenizer implementation
한국어-영어 토크나이저 구현
"""
import os
import json
import pickle
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import regex as re


class KoreanEnglishTokenizer:
    """한국어-영어 BPE 토크나이저"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
        }
        self.vocab.update(self.special_tokens)
        
        # 한국어 정규표현식 패턴
        self.korean_pattern = re.compile(r'[\uAC00-\uD7AF]+')  # 한글
        self.english_pattern = re.compile(r'[a-zA-Z]+')        # 영어
        self.number_pattern = re.compile(r'\d+')               # 숫자
        
        # BPE를 위한 패턴들
        self.pre_tokenize_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
    
    def normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 유니코드 정규화
        text = unicodedata.normalize('NFD', text)
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def pre_tokenize(self, text: str) -> List[str]:
        """사전 토크나이징 - 단어 단위로 분할"""
        text = self.normalize_text(text)
        tokens = self.pre_tokenize_pattern.findall(text)
        return [token for token in tokens if token.strip()]
    
    def get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """단어 빈도 계산"""
        word_freqs = Counter()
        
        for text in texts:
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1
        
        return dict(word_freqs)
    
    def split_word(self, word: str) -> List[str]:
        """단어를 문자 단위로 분할"""
        return list(word)
    
    def get_stats(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """연속된 토큰 쌍의 빈도 계산"""
        pairs = defaultdict(int)
        
        for word, freq in splits.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return pairs
    
    def merge_symbols(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """가장 빈번한 쌍을 병합"""
        new_splits = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in splits:
            new_word = p.sub(''.join(pair), ' '.join(splits[word]))
            new_splits[word] = new_word.split()
        
        return new_splits
    
    def train(self, texts: List[str], save_path: Optional[str] = None):
        """BPE 토크나이저 학습"""
        print("토크나이저 학습을 시작합니다...")
        
        # 1. 단어 빈도 계산
        word_freqs = self.get_word_freqs(texts)
        print(f"고유 단어 수: {len(word_freqs)}")
        
        # 2. 단어를 문자 단위로 분할
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = self.split_word(word)
        
        # 3. 초기 어휘 구축 (문자 단위)
        alphabet = set()
        for word in word_freqs:
            alphabet.update(self.split_word(word))
        
        # 특수 토큰 제외하고 어휘에 추가
        vocab_idx = len(self.special_tokens)
        for char in sorted(alphabet):
            if char not in self.vocab:
                self.vocab[char] = vocab_idx
                vocab_idx += 1
        
        # 4. BPE 병합 학습
        target_vocab_size = self.vocab_size
        merges = []
        
        while len(self.vocab) < target_vocab_size:
            # 가장 빈번한 쌍 찾기
            pairs = self.get_stats(splits)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            # 쌍 병합
            splits = self.merge_symbols(best_pair, splits)
            merges.append(best_pair)
            
            # 새로운 토큰을 어휘에 추가
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            
            if len(self.vocab) % 1000 == 0:
                print(f"어휘 크기: {len(self.vocab)}")
        
        self.merges = merges
        print(f"학습 완료! 최종 어휘 크기: {len(self.vocab)}")
        
        # 역방향 어휘 생성
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        if save_path:
            self.save(save_path)
    
    def encode_word(self, word: str) -> List[str]:
        """단어를 서브워드로 인코딩"""
        word_tokens = self.split_word(word)
        
        # BPE 병합 적용
        for pair in self.merges:
            new_word = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == pair[0] and 
                    word_tokens[i + 1] == pair[1]):
                    new_word.append(''.join(pair))
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            word_tokens = new_word
        
        return word_tokens
    
    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID로 인코딩"""
        words = self.pre_tokenize(text)
        token_ids = [self.special_tokens['<bos>']]
        
        for word in words:
            word_tokens = self.encode_word(word)
            for token in word_tokens:
                token_id = self.vocab.get(token, self.special_tokens['<unk>'])
                token_ids.append(token_id)
        
        token_ids.append(self.special_tokens['<eos>'])
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        text = ''.join(tokens)
        # 단어 경계 복원 (간단한 방법)
        text = re.sub(r'([a-zA-Z])([가-힣])', r'\1 \2', text)
        text = re.sub(r'([가-힣])([a-zA-Z])', r'\1 \2', text)
        
        return text
    
    def save(self, save_path: str):
        """토크나이저 저장"""
        os.makedirs(save_path, exist_ok=True)
        
        # 어휘 저장
        with open(os.path.join(save_path, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # 병합 규칙 저장
        with open(os.path.join(save_path, 'merges.pkl'), 'wb') as f:
            pickle.dump(self.merges, f)
        
        # 설정 저장
        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"토크나이저가 {save_path}에 저장되었습니다.")
    
    def load(self, load_path: str):
        """토크나이저 로드"""
        # 어휘 로드
        with open(os.path.join(load_path, 'vocab.json'), 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # 병합 규칙 로드
        with open(os.path.join(load_path, 'merges.pkl'), 'rb') as f:
            self.merges = pickle.load(f)
        
        # 설정 로드
        with open(os.path.join(load_path, 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.vocab_size = config['vocab_size']
            self.special_tokens = config['special_tokens']
        
        # 역방향 어휘 생성
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"토크나이저가 {load_path}에서 로드되었습니다.")
    
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return len(self.vocab)


if __name__ == "__main__":
    # 간단한 테스트
    tokenizer = KoreanEnglishTokenizer(vocab_size=1000)
    
    # 테스트 데이터
    test_texts = [
        "안녕하세요 Hello world",
        "한국어와 영어를 동시에 지원하는 토크나이저입니다.",
        "This is a Korean-English tokenizer.",
        "머신러닝과 딥러닝을 공부하고 있습니다.",
        "I am studying machine learning and deep learning."
    ]
    
    # 학습
    tokenizer.train(test_texts)
    
    # 테스트
    for text in test_texts[:2]:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"원본: {text}")
        print(f"인코딩: {encoded}")
        print(f"디코딩: {decoded}")
        print("-" * 50) 