#!/usr/bin/env python3
"""
한국어 sLLM 사전학습 데이터 전처리 모듈
Pretraining data preprocessing module for Korean sLLM

이 모듈은 다운로드된 원시 데이터를 사전학습에 적합하도록 전처리합니다.
한영 혼합 비율, 품질 필터링, 중복 제거 등을 수행합니다.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import random
from collections import defaultdict

# 프로젝트 루트 추가
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
from tqdm import tqdm
import hashlib

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess_pretraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PretrainingDataProcessor:
    """사전학습 데이터 전처리 클래스"""
    
    def __init__(self, raw_data_dir: str = "raw_datasets", output_dir: str = "datasets"):
        """
        초기화
        
        Args:
            raw_data_dir: 원시 데이터 디렉토리
            output_dir: 전처리된 데이터 저장 디렉토리
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 한영 혼합 비율 설정
        self.language_ratios = {
            "ko": 0.7,  # 한국어 70%
            "en": 0.3   # 영어 30%
        }
        
        # 텍스트 품질 필터링 설정
        self.min_text_length = 50      # 최소 텍스트 길이
        self.max_text_length = 5000    # 최대 텍스트 길이
        self.min_words_korean = 5      # 한국어 최소 단어 수
        self.min_words_english = 8     # 영어 최소 단어 수
        
        # 중복 제거용 해시셋
        self.seen_hashes = set()
        
        # 처리 통계
        self.stats = {
            "total_processed": 0,
            "korean_texts": 0,
            "english_texts": 0,
            "filtered_out": 0,
            "duplicates_removed": 0
        }
    
    def _extract_text_from_item(self, item: Dict, source_name: str) -> Optional[str]:
        """
        데이터 아이템에서 텍스트 추출
        
        Args:
            item: 데이터 아이템
            source_name: 데이터 소스 이름
            
        Returns:
            추출된 텍스트 또는 None
        """
        text = None
        
        # 소스별 텍스트 추출 로직
        if "wiki" in source_name:
            text = item.get("text", "")
        elif "news" in source_name:
            text = item.get("text", "") or item.get("content", "")
        elif "petitions" in source_name:
            title = item.get("title", "")
            content = item.get("content", "")
            text = f"{title}\n{content}" if title and content else (title or content)
        elif "chat" in source_name:
            text = item.get("text", "") or item.get("context", "")
        elif "common_crawl" in source_name or "oscar" in source_name:
            text = item.get("text", "")
        elif "openwebtext" in source_name:
            text = item.get("text", "")
        elif "c4" in source_name:
            text = item.get("text", "")
        elif "pile" in source_name:
            text = item.get("text", "")
        elif "reddit" in source_name:
            text = item.get("body", "") or item.get("text", "")
        else:
            # 일반적인 필드들 시도
            text = (item.get("text", "") or 
                   item.get("content", "") or 
                   item.get("body", "") or
                   item.get("article", ""))
        
        return text.strip() if text else None
    
    def _detect_language(self, text: str) -> str:
        """
        텍스트 언어 감지 (간단한 휴리스틱)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            언어 코드 ('ko' 또는 'en')
        """
        # 한글 문자 비율 계산
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return "en"
        
        korean_ratio = korean_chars / total_chars
        
        # 한글 비율이 20% 이상이면 한국어로 판단
        return "ko" if korean_ratio >= 0.2 else "en"
    
    def _is_high_quality_text(self, text: str, language: str) -> bool:
        """
        텍스트 품질 검사
        
        Args:
            text: 입력 텍스트
            language: 언어 코드
            
        Returns:
            고품질 여부
        """
        # 길이 검사
        if len(text) < self.min_text_length or len(text) > self.max_text_length:
            return False
        
        # 단어 수 검사
        if language == "ko":
            # 한국어: 공백과 구두점으로 분리
            words = re.findall(r'[가-힣]+', text)
            if len(words) < self.min_words_korean:
                return False
        else:
            # 영어: 공백으로 분리
            words = text.split()
            if len(words) < self.min_words_english:
                return False
        
        # 반복 패턴 검사
        if self._has_repetitive_patterns(text):
            return False
        
        # URL이 너무 많은 경우 제외
        url_count = len(re.findall(r'http[s]?://\S+', text))
        if url_count > 3:
            return False
        
        # 숫자가 너무 많은 경우 제외
        digit_ratio = len(re.findall(r'\d', text)) / len(text)
        if digit_ratio > 0.3:
            return False
        
        return True
    
    def _has_repetitive_patterns(self, text: str) -> bool:
        """
        반복 패턴 검사
        
        Args:
            text: 입력 텍스트
            
        Returns:
            반복 패턴 존재 여부
        """
        # 같은 문장이 3번 이상 반복되는 경우
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) >= 3:
            for sentence in sentences:
                if len(sentence) > 10 and text.count(sentence) >= 3:
                    return True
        
        # 같은 단어가 연속으로 5번 이상 나오는 경우
        words = text.split()
        for i in range(len(words) - 4):
            if all(words[i] == words[i+j] for j in range(5)):
                return True
        
        return False
    
    def _get_text_hash(self, text: str) -> str:
        """
        텍스트 해시 생성 (중복 제거용)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            MD5 해시
        """
        # 공백과 구두점 정규화 후 해시 생성
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, text: str) -> bool:
        """
        중복 텍스트 검사
        
        Args:
            text: 입력 텍스트
            
        Returns:
            중복 여부
        """
        text_hash = self._get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def load_raw_data(self) -> Dict[str, List[Dict]]:
        """
        원시 데이터 로드
        
        Returns:
            언어별로 그룹화된 데이터
        """
        logger.info("📂 원시 데이터 로드 시작")
        
        grouped_data = {"ko": [], "en": []}
        
        # 원시 데이터 파일들 찾기
        raw_files = list(self.raw_data_dir.glob("*.jsonl"))
        
        if not raw_files:
            logger.warning("❌ 원시 데이터 파일을 찾을 수 없습니다.")
            return grouped_data
        
        for file_path in raw_files:
            logger.info(f"처리 중: {file_path.name}")
            
            # 파일명에서 소스 이름 추출
            source_name = file_path.stem
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(tqdm(f, desc=f"로딩: {file_path.name}")):
                        try:
                            item = json.loads(line.strip())
                            
                            # 텍스트 추출
                            text = self._extract_text_from_item(item, source_name)
                            if not text:
                                continue
                            
                            # 언어 감지
                            language = item.get('language') or self._detect_language(text)
                            
                            # 품질 검사
                            if not self._is_high_quality_text(text, language):
                                self.stats["filtered_out"] += 1
                                continue
                            
                            # 중복 검사
                            if self._is_duplicate(text):
                                self.stats["duplicates_removed"] += 1
                                continue
                            
                            # 데이터 추가
                            processed_item = {
                                "text": text,
                                "language": language,
                                "source": source_name,
                                "length": len(text)
                            }
                            
                            grouped_data[language].append(processed_item)
                            self.stats["total_processed"] += 1
                            
                            if language == "ko":
                                self.stats["korean_texts"] += 1
                            else:
                                self.stats["english_texts"] += 1
                                
                        except json.JSONDecodeError:
                            logger.warning(f"JSON 파싱 오류: {file_path.name}:{line_num}")
                            continue
                        except Exception as e:
                            logger.warning(f"처리 오류: {file_path.name}:{line_num} - {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"파일 읽기 오류: {file_path.name} - {e}")
                continue
        
        logger.info(f"✅ 원시 데이터 로드 완료")
        logger.info(f"한국어: {len(grouped_data['ko'])}개, 영어: {len(grouped_data['en'])}개")
        
        return grouped_data
    
    def create_mixed_dataset(self, grouped_data: Dict[str, List[Dict]], 
                           target_total_size: Optional[int] = None) -> List[Dict]:
        """
        한영 혼합 데이터셋 생성
        
        Args:
            grouped_data: 언어별로 그룹화된 데이터
            target_total_size: 목표 데이터 크기 (None이면 전체 사용)
            
        Returns:
            혼합된 데이터셋
        """
        logger.info("🔀 한영 혼합 데이터셋 생성 시작")
        
        korean_data = grouped_data["ko"]
        english_data = grouped_data["en"]
        
        # 목표 크기 설정
        if target_total_size is None:
            target_total_size = len(korean_data) + len(english_data)
        
        # 언어별 목표 크기 계산
        target_korean = int(target_total_size * self.language_ratios["ko"])
        target_english = int(target_total_size * self.language_ratios["en"])
        
        logger.info(f"목표 크기 - 한국어: {target_korean}, 영어: {target_english}")
        
        # 데이터 샘플링
        if len(korean_data) > target_korean:
            korean_sample = random.sample(korean_data, target_korean)
        else:
            korean_sample = korean_data
            logger.warning(f"한국어 데이터 부족: {len(korean_data)} < {target_korean}")
        
        if len(english_data) > target_english:
            english_sample = random.sample(english_data, target_english)
        else:
            english_sample = english_data
            logger.warning(f"영어 데이터 부족: {len(english_data)} < {target_english}")
        
        # 혼합 및 셔플
        mixed_data = korean_sample + english_sample
        random.shuffle(mixed_data)
        
        logger.info(f"✅ 혼합 데이터셋 생성 완료: {len(mixed_data)}개")
        
        return mixed_data
    
    def save_pretraining_dataset(self, dataset: List[Dict], 
                                dataset_type: str = "mixed") -> str:
        """
        사전학습 데이터셋 저장
        
        Args:
            dataset: 데이터셋
            dataset_type: 데이터셋 타입 (mixed, korean, english)
            
        Returns:
            저장된 파일 경로
        """
        output_file = self.output_dir / f"{dataset_type}_pretraining_corpus.json"
        
        logger.info(f"💾 데이터셋 저장 중: {output_file}")
        
        # 텍스트만 추출하여 저장 (사전학습용 형식)
        corpus = []
        for item in tqdm(dataset, desc="데이터셋 변환"):
            corpus.append({
                "text": item["text"],
                "language": item["language"],
                "source": item["source"],
                "length": item["length"]
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 저장 완료: {output_file}")
        return str(output_file)
    
    def save_statistics(self):
        """처리 통계 저장"""
        stats_file = self.output_dir / "pretraining_preprocessing_stats.json"
        
        # 언어별 비율 계산
        if self.stats["total_processed"] > 0:
            korean_ratio = self.stats["korean_texts"] / self.stats["total_processed"]
            english_ratio = self.stats["english_texts"] / self.stats["total_processed"]
        else:
            korean_ratio = english_ratio = 0
        
        detailed_stats = {
            **self.stats,
            "language_ratios": {
                "korean": round(korean_ratio, 3),
                "english": round(english_ratio, 3)
            },
            "processing_time": datetime.now().isoformat(),
            "quality_filters": {
                "min_text_length": self.min_text_length,
                "max_text_length": self.max_text_length,
                "min_words_korean": self.min_words_korean,
                "min_words_english": self.min_words_english
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 통계 저장 완료: {stats_file}")
    
    def get_processing_summary(self) -> str:
        """처리 요약 정보 반환"""
        korean_ratio = (self.stats["korean_texts"] / self.stats["total_processed"] 
                       if self.stats["total_processed"] > 0 else 0)
        english_ratio = (self.stats["english_texts"] / self.stats["total_processed"] 
                        if self.stats["total_processed"] > 0 else 0)
        
        summary = f"""
📊 사전학습 데이터 전처리 완료 요약
==================================
총 처리된 텍스트: {self.stats['total_processed']:,}개
├─ 한국어: {self.stats['korean_texts']:,}개 ({korean_ratio:.1%})
└─ 영어: {self.stats['english_texts']:,}개 ({english_ratio:.1%})

필터링 결과:
├─ 품질 필터로 제외: {self.stats['filtered_out']:,}개
└─ 중복 제거: {self.stats['duplicates_removed']:,}개

목표 언어 비율:
├─ 한국어: {self.language_ratios['ko']:.1%}
└─ 영어: {self.language_ratios['en']:.1%}

저장 위치: {self.output_dir}
"""
        return summary


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="사전학습 데이터 전처리")
    
    parser.add_argument(
        "--raw-data-dir",
        default="raw_datasets", 
        help="원시 데이터 디렉토리 (기본값: raw_datasets)"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="출력 디렉토리 (기본값: datasets)"
    )
    parser.add_argument(
        "--korean-ratio",
        type=float,
        default=0.7,
        help="한국어 비율 (기본값: 0.7)"
    )
    parser.add_argument(
        "--english-ratio", 
        type=float,
        default=0.3,
        help="영어 비율 (기본값: 0.3)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        help="목표 데이터셋 크기 (지정하지 않으면 전체 사용)"
    )
    parser.add_argument(
        "--korean-only",
        action="store_true",
        help="한국어 전용 데이터셋만 생성"
    )
    parser.add_argument(
        "--english-only", 
        action="store_true",
        help="영어 전용 데이터셋만 생성"
    )
    parser.add_argument(
        "--mixed-only",
        action="store_true", 
        help="혼합 데이터셋만 생성"
    )
    
    args = parser.parse_args()
    
    # 언어 비율 검증
    if abs((args.korean_ratio + args.english_ratio) - 1.0) > 0.01:
        logger.error("❌ 한국어와 영어 비율의 합이 1.0이 되어야 합니다.")
        sys.exit(1)
    
    # 전처리기 초기화
    processor = PretrainingDataProcessor(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir
    )
    
    # 언어 비율 설정
    processor.language_ratios = {
        "ko": args.korean_ratio,
        "en": args.english_ratio
    }
    
    try:
        # 원시 데이터 로드
        grouped_data = processor.load_raw_data()
        
        if not grouped_data["ko"] and not grouped_data["en"]:
            logger.error("❌ 처리할 데이터가 없습니다.")
            sys.exit(1)
        
        # 데이터셋 생성 및 저장
        if args.korean_only or (not args.english_only and not args.mixed_only):
            if grouped_data["ko"]:
                logger.info("🇰🇷 한국어 전용 데이터셋 생성")
                processor.save_pretraining_dataset(grouped_data["ko"], "korean")
        
        if args.english_only or (not args.korean_only and not args.mixed_only):
            if grouped_data["en"]:
                logger.info("🇺🇸 영어 전용 데이터셋 생성")
                processor.save_pretraining_dataset(grouped_data["en"], "english")
        
        if args.mixed_only or (not args.korean_only and not args.english_only):
            if grouped_data["ko"] and grouped_data["en"]:
                logger.info("🔀 혼합 데이터셋 생성")
                mixed_dataset = processor.create_mixed_dataset(
                    grouped_data, 
                    target_total_size=args.target_size
                )
                processor.save_pretraining_dataset(mixed_dataset, "mixed")
        
        # 통계 저장
        processor.save_statistics()
        
        # 요약 정보 출력
        print(processor.get_processing_summary())
        
        logger.info("✅ 사전학습 데이터 전처리 완료!")
        
    except Exception as e:
        logger.error(f"❌ 전처리 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 