#!/usr/bin/env python3
"""
한국어 sLLM 원시 데이터 다운로드 모듈
Raw data download module for Korean sLLM

이 모듈은 순수하게 원시 데이터 다운로드만 담당합니다.
전처리는 별도 모듈에서 수행됩니다.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# 프로젝트 루트 추가
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset
import requests
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_raw_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RawDataDownloader:
    """원시 데이터 다운로드 클래스"""
    
    def __init__(self, output_dir: str = "raw_datasets"):
        """
        초기화
        
        Args:
            output_dir: 원시 데이터 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드 메타데이터 저장
        self.metadata = {
            "download_time": datetime.now().isoformat(),
            "datasets": {}
        }
    
    def download_korean_datasets(self, small_sample: bool = False) -> Dict[str, str]:
        """
        한국어 원시 데이터셋 다운로드
        
        Args:
            small_sample: 테스트용 소량 샘플만 다운로드할지 여부
            
        Returns:
            다운로드된 파일 경로들
        """
        logger.info("🇰🇷 한국어 원시 데이터셋 다운로드 시작")
        
        korean_sources = [
            {
                "name": "kowiki",
                "dataset_id": "wikipedia",
                "config": "20231101.ko",
                "split": "train",
                "description": "한국어 위키피디아"
            },
            {
                "name": "korean_news",
                "dataset_id": "klue",
                "config": "ynat",
                "split": "train", 
                "description": "한국어 뉴스 분류 데이터"
            },
            {
                "name": "korean_petitions",
                "dataset_id": "heegyu/petitions",
                "config": None,
                "split": "train",
                "description": "청와대 국민청원 데이터"
            },
            {
                "name": "korean_chat",
                "dataset_id": "heegyu/korquad-chat-v1",
                "config": None,
                "split": "train",
                "description": "한국어 대화 데이터"
            },
            {
                "name": "korean_common_crawl",
                "dataset_id": "oscar-corpus/OSCAR-2301",
                "config": "ko",
                "split": "train",
                "description": "한국어 Common Crawl 데이터"
            }
        ]
        
        downloaded_files = {}
        
        for source in korean_sources:
            try:
                logger.info(f"다운로드 중: {source['description']}")
                
                # 데이터셋 로드
                if source['config']:
                    dataset = load_dataset(
                        source['dataset_id'], 
                        source['config'], 
                        split=source['split'],
                        streaming=True if not small_sample else False
                    )
                else:
                    dataset = load_dataset(
                        source['dataset_id'], 
                        split=source['split'],
                        streaming=True if not small_sample else False
                    )
                
                # 샘플링 (테스트용)
                if small_sample:
                    if hasattr(dataset, 'select'):
                        dataset = dataset.select(range(min(1000, len(dataset))))
                    else:
                        # 스트리밍 데이터셋의 경우
                        dataset = dataset.take(1000)
                
                # 원시 데이터 저장
                output_file = self.output_dir / f"korean_raw_{source['name']}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for i, item in enumerate(tqdm(dataset, desc=f"저장 중: {source['name']}")):
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                        
                        # 소량 샘플의 경우 제한
                        if small_sample and i >= 999:
                            break
                
                downloaded_files[source['name']] = str(output_file)
                
                # 메타데이터 업데이트
                self.metadata["datasets"][source['name']] = {
                    "source": source,
                    "file_path": str(output_file),
                    "file_size": output_file.stat().st_size,
                    "small_sample": small_sample
                }
                
                logger.info(f"✅ {source['name']} 다운로드 완료: {output_file}")
                
            except Exception as e:
                logger.error(f"❌ {source['name']} 다운로드 실패: {e}")
                continue
        
        return downloaded_files
    
    def download_english_datasets(self, small_sample: bool = False) -> Dict[str, str]:
        """
        영어 원시 데이터셋 다운로드
        
        Args:
            small_sample: 테스트용 소량 샘플만 다운로드할지 여부
            
        Returns:
            다운로드된 파일 경로들
        """
        logger.info("🇺🇸 영어 원시 데이터셋 다운로드 시작")
        
        english_sources = [
            {
                "name": "enwiki",
                "dataset_id": "wikipedia",
                "config": "20231101.en", 
                "split": "train",
                "description": "영어 위키피디아"
            },
            {
                "name": "openwebtext",
                "dataset_id": "openwebtext",
                "config": None,
                "split": "train",
                "description": "OpenWebText 데이터"
            },
            {
                "name": "c4_en",
                "dataset_id": "c4",
                "config": "en",
                "split": "train",
                "description": "C4 영어 데이터"
            },
            {
                "name": "pile_subset",
                "dataset_id": "EleutherAI/pile",
                "config": None,
                "split": "train",
                "description": "The Pile 데이터셋"
            },
            {
                "name": "reddit_comments",
                "dataset_id": "reddit",
                "config": None,
                "split": "train", 
                "description": "Reddit 댓글 데이터"
            }
        ]
        
        downloaded_files = {}
        
        for source in english_sources:
            try:
                logger.info(f"다운로드 중: {source['description']}")
                
                # 데이터셋 로드
                if source['config']:
                    dataset = load_dataset(
                        source['dataset_id'],
                        source['config'],
                        split=source['split'],
                        streaming=True if not small_sample else False
                    )
                else:
                    dataset = load_dataset(
                        source['dataset_id'],
                        split=source['split'], 
                        streaming=True if not small_sample else False
                    )
                
                # 샘플링 (테스트용)
                if small_sample:
                    if hasattr(dataset, 'select'):
                        dataset = dataset.select(range(min(1000, len(dataset))))
                    else:
                        # 스트리밍 데이터셋의 경우
                        dataset = dataset.take(1000)
                
                # 원시 데이터 저장
                output_file = self.output_dir / f"english_raw_{source['name']}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for i, item in enumerate(tqdm(dataset, desc=f"저장 중: {source['name']}")):
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                        
                        # 소량 샘플의 경우 제한
                        if small_sample and i >= 999:
                            break
                
                downloaded_files[source['name']] = str(output_file)
                
                # 메타데이터 업데이트
                self.metadata["datasets"][source['name']] = {
                    "source": source,
                    "file_path": str(output_file),
                    "file_size": output_file.stat().st_size,
                    "small_sample": small_sample
                }
                
                logger.info(f"✅ {source['name']} 다운로드 완료: {output_file}")
                
            except Exception as e:
                logger.error(f"❌ {source['name']} 다운로드 실패: {e}")
                continue
        
        return downloaded_files
    
    def download_instruction_datasets(self, small_sample: bool = False) -> Dict[str, str]:
        """
        미세조정용 명령어 데이터셋 다운로드
        
        Args:
            small_sample: 테스트용 소량 샘플만 다운로드할지 여부
            
        Returns:
            다운로드된 파일 경로들
        """
        logger.info("🎯 미세조정용 명령어 데이터셋 다운로드 시작")
        
        instruction_sources = [
            # 한국어 명령어 데이터
            {
                "name": "korean_alpaca",
                "dataset_id": "beomi/KoAlpaca-v1.1a",
                "config": None,
                "split": "train",
                "description": "한국어 Alpaca 데이터",
                "language": "ko"
            },
            {
                "name": "korean_chatgpt",
                "dataset_id": "FreedomIntelligence/evol-instruct-korean",
                "config": None,
                "split": "train", 
                "description": "한국어 ChatGPT 스타일 데이터",
                "language": "ko"
            },
            {
                "name": "korean_sharegpt",
                "dataset_id": "maywell/ko_Ultrachat_200k",
                "config": None,
                "split": "train",
                "description": "한국어 ShareGPT 데이터",
                "language": "ko"
            },
            # 영어 명령어 데이터
            {
                "name": "alpaca_english",
                "dataset_id": "tatsu-lab/alpaca",
                "config": None,
                "split": "train",
                "description": "영어 Alpaca 데이터",
                "language": "en"
            },
            {
                "name": "dolly_english", 
                "dataset_id": "databricks/databricks-dolly-15k",
                "config": None,
                "split": "train",
                "description": "Dolly 15k 데이터",
                "language": "en"
            },
            {
                "name": "oasst1_english",
                "dataset_id": "OpenAssistant/oasst1",
                "config": None,
                "split": "train",
                "description": "OpenAssistant 데이터",
                "language": "en"
            },
            {
                "name": "ultrachat_english",
                "dataset_id": "stingning/ultrachat",
                "config": None,
                "split": "train",
                "description": "UltraChat 데이터",
                "language": "en"
            }
        ]
        
        downloaded_files = {}
        
        for source in instruction_sources:
            try:
                logger.info(f"다운로드 중: {source['description']}")
                
                # 데이터셋 로드
                if source['config']:
                    dataset = load_dataset(
                        source['dataset_id'],
                        source['config'],
                        split=source['split'],
                        streaming=True if not small_sample else False
                    )
                else:
                    dataset = load_dataset(
                        source['dataset_id'],
                        split=source['split'],
                        streaming=True if not small_sample else False
                    )
                
                # 샘플링 (테스트용)
                if small_sample:
                    if hasattr(dataset, 'select'):
                        dataset = dataset.select(range(min(100, len(dataset))))
                    else:
                        # 스트리밍 데이터셋의 경우
                        dataset = dataset.take(100)
                
                # 원시 데이터 저장
                output_file = self.output_dir / f"instruction_raw_{source['name']}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for i, item in enumerate(tqdm(dataset, desc=f"저장 중: {source['name']}")):
                        # 언어 태그 추가
                        item['language'] = source['language']
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                        
                        # 소량 샘플의 경우 제한
                        if small_sample and i >= 99:
                            break
                
                downloaded_files[source['name']] = str(output_file)
                
                # 메타데이터 업데이트
                self.metadata["datasets"][source['name']] = {
                    "source": source,
                    "file_path": str(output_file),
                    "file_size": output_file.stat().st_size,
                    "small_sample": small_sample,
                    "language": source['language']
                }
                
                logger.info(f"✅ {source['name']} 다운로드 완료: {output_file}")
                
            except Exception as e:
                logger.error(f"❌ {source['name']} 다운로드 실패: {e}")
                continue
        
        return downloaded_files
    
    def save_metadata(self):
        """다운로드 메타데이터 저장"""
        metadata_file = self.output_dir / "download_metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 메타데이터 저장 완료: {metadata_file}")
    
    def get_download_summary(self) -> str:
        """다운로드 요약 정보 반환"""
        total_files = len(self.metadata["datasets"])
        total_size = sum(
            dataset_info["file_size"] 
            for dataset_info in self.metadata["datasets"].values()
        )
        
        summary = f"""
📊 다운로드 완료 요약
==================
총 파일 수: {total_files}개
총 용량: {total_size / (1024**3):.2f} GB
저장 위치: {self.output_dir}

파일 목록:
"""
        
        for name, info in self.metadata["datasets"].items():
            file_size_mb = info["file_size"] / (1024**2)
            summary += f"  - {name}: {file_size_mb:.1f} MB\n"
        
        return summary


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="한국어 sLLM 원시 데이터 다운로드")
    
    parser.add_argument(
        "--output-dir", 
        default="raw_datasets",
        help="원시 데이터 저장 디렉토리 (기본값: raw_datasets)"
    )
    parser.add_argument(
        "--korean", 
        action="store_true",
        help="한국어 데이터셋만 다운로드"
    )
    parser.add_argument(
        "--english", 
        action="store_true",
        help="영어 데이터셋만 다운로드"
    )
    parser.add_argument(
        "--instructions", 
        action="store_true",
        help="명령어 데이터셋만 다운로드"
    )
    parser.add_argument(
        "--small", 
        action="store_true",
        help="테스트용 소량 샘플만 다운로드"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="모든 데이터셋 다운로드"
    )
    
    args = parser.parse_args()
    
    # 다운로더 초기화
    downloader = RawDataDownloader(output_dir=args.output_dir)
    
    try:
        # 다운로드 실행
        if args.all or (not args.korean and not args.english and not args.instructions):
            logger.info("🚀 모든 데이터셋 다운로드 시작")
            downloader.download_korean_datasets(small_sample=args.small)
            downloader.download_english_datasets(small_sample=args.small)
            downloader.download_instruction_datasets(small_sample=args.small)
        else:
            if args.korean:
                downloader.download_korean_datasets(small_sample=args.small)
            if args.english:
                downloader.download_english_datasets(small_sample=args.small)
            if args.instructions:
                downloader.download_instruction_datasets(small_sample=args.small)
        
        # 메타데이터 저장
        downloader.save_metadata()
        
        # 요약 정보 출력
        print(downloader.get_download_summary())
        
        logger.info("✅ 모든 다운로드 작업 완료!")
        
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 