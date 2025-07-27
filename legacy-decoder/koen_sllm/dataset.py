"""
Dataset downloading and preprocessing for Korean sLLM
한국어 sLLM을 위한 데이터셋 다운로드 및 전처리
"""
import os
import json
import torch
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import logging
from tqdm import tqdm

from .tokenizer import KoreanEnglishTokenizer
from .config import ModelConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """다양한 소스에서 데이터셋을 다운로드하는 클래스"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_korean_datasets(self):
        """한국어 데이터셋 다운로드"""
        logger.info("한국어 데이터셋 다운로드 시작...")
        
        korean_datasets = []
        
        try:
            # 1. AI Hub의 일상대화 데이터
            logger.info("AI Hub 일상대화 데이터 로드 중...")
            aihub_casual = load_dataset("lcw99/wikipedia-korean-20240501", cache_dir="models")
            korean_datasets.append(aihub_casual)
        except Exception as e:
            logger.warning(f"AI Hub 데이터 로드 실패: {e}")
        
        try:
            # 2. 한국어 위키피디아
            logger.info("한국어 위키피디아 로드 중...")
            wiki_ko = load_dataset("wikipedia", "20220301.ko", split="train", cache_dir="models")
            korean_datasets.append(wiki_ko)
        except Exception as e:
            logger.warning(f"한국어 위키피디아 로드 실패: {e}")
        
        try:
            # 3. KLUE 뉴스 데이터
            logger.info("KLUE 데이터 로드 중...")
            klue = load_dataset("klue", "ynat", split="train", cache_dir="models")
            korean_datasets.append(klue)
        except Exception as e:
            logger.warning(f"KLUE 데이터 로드 실패: {e}")
        
        try:
            # 4. 한국어 CommonCrawl 데이터
            logger.info("한국어 CommonCrawl 데이터 로드 중...")
            cc_ko = load_dataset("oscar", "unshuffled_deduplicated_ko", split="train", streaming=True, cache_dir="models")
            # 스트리밍 데이터에서 일부만 가져오기
            cc_ko_sample = []
            for i, example in enumerate(cc_ko):
                if i >= 50000:  # 5만개 샘플
                    break
                cc_ko_sample.append(example)
            korean_datasets.append(cc_ko_sample)
        except Exception as e:
            logger.warning(f"한국어 CommonCrawl 로드 실패: {e}")
        
        # 한국어 데이터 저장
        korean_data = self._merge_datasets(korean_datasets, "korean")
        self._save_dataset(korean_data, "korean_corpus.json")
        
        return korean_data
    
    def download_english_datasets(self):
        """영어 데이터셋 다운로드"""
        logger.info("영어 데이터셋 다운로드 시작...")
        
        english_datasets = []
        
        try:
            # 1. OpenWebText
            logger.info("OpenWebText 로드 중...")
            openwebtext = load_dataset("openwebtext", split="train", streaming=True, cache_dir="models")
            # 스트리밍에서 일부만 가져오기
            owt_sample = []
            for i, example in enumerate(openwebtext):
                if i >= 100000:  # 10만개 샘플
                    break
                owt_sample.append(example)
            english_datasets.append(owt_sample)
        except Exception as e:
            logger.warning(f"OpenWebText 로드 실패: {e}")
        
        try:
            # 2. WikiText-103
            logger.info("WikiText-103 로드 중...")
            wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir="models")
            english_datasets.append(wikitext)
        except Exception as e:
            logger.warning(f"WikiText 로드 실패: {e}")
        
        try:
            # 3. BookCorpus (대체: gutenberg books)
            logger.info("Gutenberg books 로드 중...")
            books = load_dataset("sedthh/gutenberg_english", split="train", cache_dir="models")
            # 샘플링
            books_sample = books.select(range(min(10000, len(books))))
            english_datasets.append(books_sample)
        except Exception as e:
            logger.warning(f"Books 데이터 로드 실패: {e}")
        
        try:
            # 4. CC-News 영어
            logger.info("CC-News 영어 로드 중...")
            cc_news = load_dataset("cc_news", split="train", streaming=True, cache_dir="models")
            news_sample = []
            for i, example in enumerate(cc_news):
                if i >= 50000:  # 5만개 샘플
                    break
                if self._is_english_text(example.get('text', '')):
                    news_sample.append(example)
            english_datasets.append(news_sample)
        except Exception as e:
            logger.warning(f"CC-News 로드 실패: {e}")
        
        # 영어 데이터 저장
        english_data = self._merge_datasets(english_datasets, "english")
        self._save_dataset(english_data, "english_corpus.json")
        
        return english_data
    
    def _is_english_text(self, text: str) -> bool:
        """텍스트가 영어인지 간단히 판단"""
        if not text or len(text) < 10:
            return False
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        return total_chars > 0 and (english_chars / total_chars) > 0.8
    
    def _merge_datasets(self, datasets: List, language: str) -> List[str]:
        """데이터셋들을 병합하여 텍스트 리스트로 변환"""
        merged_texts = []
        
        for dataset in datasets:
            if isinstance(dataset, list):
                # 딕셔너리 리스트인 경우
                for item in dataset:
                    text = self._extract_text(item)
                    if text and len(text.strip()) > 50:  # 최소 길이 필터
                        merged_texts.append(text.strip())
            else:
                # HuggingFace Dataset 객체인 경우
                for item in dataset:
                    text = self._extract_text(item)
                    if text and len(text.strip()) > 50:
                        merged_texts.append(text.strip())
        
        logger.info(f"{language} 데이터: {len(merged_texts)}개 문서")
        return merged_texts
    
    def _extract_text(self, item: Dict) -> Optional[str]:
        """데이터 아이템에서 텍스트 추출"""
        # 다양한 필드명에서 텍스트 추출 시도
        text_fields = ['text', 'content', 'article', 'document', 'sentence', 'title']
        
        for field in text_fields:
            if field in item and item[field]:
                return str(item[field])
        
        # 딕셔너리의 첫 번째 문자열 값 사용
        for value in item.values():
            if isinstance(value, str) and len(value) > 20:
                return value
        
        return None
    
    def _save_dataset(self, texts: List[str], filename: str):
        """데이터셋을 JSON 파일로 저장"""
        save_path = self.data_dir / filename
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"데이터셋이 {save_path}에 저장되었습니다.")
    
    def download_all(self) -> Tuple[List[str], List[str]]:
        """모든 데이터셋 다운로드"""
        korean_data = self.download_korean_datasets()
        english_data = self.download_english_datasets()
        return korean_data, english_data


class TextDataset(Dataset):
    """텍스트 데이터셋 클래스"""
    
    def __init__(self, 
                 tokenizer: KoreanEnglishTokenizer,
                 texts: List[str],
                 max_length: int = 512,
                 stride: int = 256):
        """
        Args:
            tokenizer: 토크나이저
            texts: 텍스트 리스트
            max_length: 최대 시퀀스 길이
            stride: 슬라이딩 윈도우 스트라이드
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # 텍스트를 토큰화하고 청크로 분할
        self.examples = []
        self._prepare_examples(texts)
        
    def _prepare_examples(self, texts: List[str]):
        """텍스트를 토큰화하고 학습 예제로 변환"""
        logger.info("텍스트 토큰화 및 청크 분할 중...")
        
        for text in tqdm(texts, desc="Processing texts"):
            # 텍스트 토큰화
            token_ids = self.tokenizer.encode(text)
            
            # 너무 짧은 텍스트는 스킵
            if len(token_ids) < 50:
                continue
            
            # 슬라이딩 윈도우로 청크 생성
            for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                chunk = token_ids[i:i + self.max_length]
                
                # 패딩 또는 자르기
                if len(chunk) < self.max_length:
                    # 패딩
                    chunk.extend([self.tokenizer.special_tokens['<pad>']] * 
                                (self.max_length - len(chunk)))
                
                self.examples.append(chunk)
        
        logger.info(f"총 {len(self.examples)}개의 학습 예제 생성")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # 입력과 라벨 (다음 토큰 예측)
        input_ids = torch.tensor(example, dtype=torch.long)
        labels = input_ids.clone()
        
        # 패딩 토큰은 라벨에서 -100으로 마스킹 (손실 계산 시 무시)
        pad_token_id = self.tokenizer.special_tokens['<pad>']
        labels[labels == pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class DatasetManager:
    """데이터셋 관리 클래스"""
    
    def __init__(self, config: ModelConfig, tokenizer: KoreanEnglishTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.downloader = DatasetDownloader()
    
    def prepare_datasets(self, download_fresh: bool = False) -> Tuple[DataLoader, DataLoader]:
        """학습 및 검증 데이터셋 준비"""
        
        # 기존 데이터가 있는지 확인
        korean_path = Path("datasets/korean_corpus.json")
        english_path = Path("datasets/english_corpus.json")
        
        if download_fresh or not (korean_path.exists() and english_path.exists()):
            logger.info("새로운 데이터셋 다운로드...")
            korean_texts, english_texts = self.downloader.download_all()
        else:
            logger.info("기존 데이터셋 로드...")
            korean_texts = self._load_texts(korean_path)
            english_texts = self._load_texts(english_path)
        
        # 데이터 결합 및 분할
        all_texts = korean_texts + english_texts
        
        # 셔플
        import random
        random.seed(self.config.seed)
        random.shuffle(all_texts)
        
        # 학습/검증 분할 (90:10)
        split_idx = int(0.9 * len(all_texts))
        train_texts = all_texts[:split_idx]
        val_texts = all_texts[split_idx:]
        
        logger.info(f"학습 데이터: {len(train_texts)}개 문서")
        logger.info(f"검증 데이터: {len(val_texts)}개 문서")
        
        # 데이터셋 객체 생성
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            texts=train_texts,
            max_length=self.config.max_seq_len,
            stride=self.config.max_seq_len // 2
        )
        
        val_dataset = TextDataset(
            tokenizer=self.tokenizer,
            texts=val_texts,
            max_length=self.config.max_seq_len,
            stride=self.config.max_seq_len // 2
        )
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def _load_texts(self, path: Path) -> List[str]:
        """JSON 파일에서 텍스트 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_tokenizer_training_data(self) -> List[str]:
        """토크나이저 학습용 데이터 준비"""
        logger.info("토크나이저 학습용 데이터 준비...")
        
        # 소량의 다양한 데이터 수집
        sample_texts = []
        
        try:
            # 한국어 샘플
            korean_wiki = load_dataset("wikipedia", "20220301.ko", split="train", cache_dir="models")
            for i, example in enumerate(korean_wiki):
                if i >= 5000:  # 5천개 샘플
                    break
                text = example.get('text', '')
                if len(text) > 100:
                    sample_texts.append(text)
            
            # 영어 샘플
            english_wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir="models")
            for i, example in enumerate(english_wiki):
                if i >= 5000:  # 5천개 샘플
                    break
                text = example.get('text', '')
                if len(text) > 100:
                    sample_texts.append(text)
                    
        except Exception as e:
            logger.warning(f"토크나이저 학습 데이터 로드 실패: {e}")
            # 기본 샘플 텍스트 사용
            sample_texts = [
                "안녕하세요. 한국어 텍스트입니다.",
                "Hello, this is English text.",
                "머신러닝과 딥러닝을 공부하고 있습니다.",
                "I am studying machine learning and deep learning.",
                "자연어처리는 정말 흥미로운 분야입니다.",
                "Natural language processing is a fascinating field."
            ]
        
        logger.info(f"토크나이저 학습용 데이터: {len(sample_texts)}개 문서")
        return sample_texts


if __name__ == "__main__":
    # 데이터셋 다운로드 테스트
    downloader = DatasetDownloader()
    
    # 토크나이저 학습용 데이터 준비
    manager = DatasetManager(ModelConfig(), None)
    training_texts = manager.prepare_tokenizer_training_data()
    
    print(f"토크나이저 학습용 데이터: {len(training_texts)}개")
    print("첫 번째 샘플:", training_texts[0][:200]) 