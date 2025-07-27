#!/usr/bin/env python3
"""
Pretraining Dataset Download Script
사전훈련용 데이터셋 다운로드 스크립트 (configs/dataset/pretraining.json 기반)
"""
import sys
import json
import argparse
import logging
from pathlib import Path

# 현재 스크립트 디렉토리
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from dataset_loader import DynamicDatasetLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PretrainingDatasetDownloader:
    """사전훈련용 데이터셋 다운로더 (configs/dataset 기반)"""
    
    def __init__(self, config_path: str = None, use_unique_names: bool = False):
        if config_path is None:
            config_path = script_dir.parent.parent / "configs" / "dataset" / "pretraining.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.use_unique_names = use_unique_names
        
        # 프로젝트 루트 계산
        self.project_root = script_dir.parent.parent.parent.parent
        self.output_dir = self.project_root / "datasets"
        self.cache_dir = self.project_root / "models"
        
        logger.info(f"📋 설정 파일: {self.config_path}")
        logger.info(f"📁 출력 디렉토리: {self.output_dir}")
        logger.info(f"💾 캐시 디렉토리: {self.cache_dir}")
        if self.use_unique_names:
            logger.info(f"📝 고유한 파일명 사용: 활성화")
        
    def _load_config(self):
        """설정 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _resolve_dataset_config_path(self, relative_path: str) -> Path:
        """상대 경로를 절대 경로로 변환"""
        base_dir = self.config_path.parent
        return (base_dir / relative_path).resolve()
    
    def get_dataset_info(self):
        """데이터셋 정보 표시"""
        logger.info(f"📊 사전훈련 데이터셋 설정:")
        logger.info(f"   설명: {self.config.get('description', 'N/A')}")
        
        # 한국어 데이터셋 정보
        if 'korean_config' in self.config:
            korean_config_path = self._resolve_dataset_config_path(self.config['korean_config'])
            if korean_config_path.exists():
                logger.info(f"\n🇰🇷 한국어 데이터셋:")
                korean_loader = DynamicDatasetLoader(str(korean_config_path), str(self.cache_dir))
                korean_info = korean_loader.get_dataset_info()
                for i, ds in enumerate(korean_info['enabled_datasets'], 1):
                    logger.info(f"   {i}. {ds['description']}")
            else:
                logger.warning(f"❌ 한국어 설정 파일 없음: {korean_config_path}")
        
        # 영어 데이터셋 정보
        if 'english_config' in self.config:
            english_config_path = self._resolve_dataset_config_path(self.config['english_config'])
            if english_config_path.exists():
                logger.info(f"\n🇺🇸 영어 데이터셋:")
                english_loader = DynamicDatasetLoader(str(english_config_path), str(self.cache_dir))
                english_info = english_loader.get_dataset_info()
                for i, ds in enumerate(english_info['enabled_datasets'], 1):
                    logger.info(f"   {i}. {ds['description']}")
            else:
                logger.warning(f"❌ 영어 설정 파일 없음: {english_config_path}")
        
        # 혼합 비율 정보
        if 'mixing_ratio' in self.config:
            logger.info(f"\n⚖️  언어별 혼합 비율:")
            for lang, ratio in self.config['mixing_ratio'].items():
                logger.info(f"   {lang}: {ratio*100:.1f}%")
    
    def download_datasets(self, small_mode: bool = False, force: bool = False):
        """사전훈련 데이터셋 다운로드"""
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        korean_data = []
        english_data = []
        
        # 한국어 데이터 다운로드
        if 'korean_config' in self.config:
            korean_config_path = self._resolve_dataset_config_path(self.config['korean_config'])
            if korean_config_path.exists():
                logger.info("🇰🇷 한국어 사전훈련 데이터 다운로드 중...")
                
                korean_loader = DynamicDatasetLoader(str(korean_config_path), str(self.cache_dir))
                korean_output_file = self.output_dir / "korean_pretraining_corpus.json"
                
                if korean_output_file.exists() and not force:
                    logger.info(f"기존 한국어 데이터 사용: {korean_output_file}")
                else:
                    if force and korean_output_file.exists():
                        korean_output_file.unlink()
                        logger.info("기존 한국어 데이터 삭제")
                    
                    korean_data = korean_loader.load_all_datasets(small_mode=small_mode)
                    if korean_data:
                        korean_loader.save_texts(korean_data, str(korean_output_file))
            else:
                logger.warning(f"한국어 설정 파일 없음: {korean_config_path}")
        
        # 영어 데이터 다운로드
        if 'english_config' in self.config:
            english_config_path = self._resolve_dataset_config_path(self.config['english_config'])
            if english_config_path.exists():
                logger.info("🇺🇸 영어 사전훈련 데이터 다운로드 중...")
                
                english_loader = DynamicDatasetLoader(str(english_config_path), str(self.cache_dir))
                english_output_file = self.output_dir / "english_pretraining_corpus.json"
                
                if english_output_file.exists() and not force:
                    logger.info(f"기존 영어 데이터 사용: {english_output_file}")
                else:
                    if force and english_output_file.exists():
                        english_output_file.unlink()
                        logger.info("기존 영어 데이터 삭제")
                    
                    english_data = english_loader.load_all_datasets(small_mode=small_mode)
                    if english_data:
                        english_loader.save_texts(english_data, str(english_output_file))
            else:
                logger.warning(f"영어 설정 파일 없음: {english_config_path}")
        
        # 혼합 데이터셋 생성
        if korean_data or english_data:
            self._create_mixed_dataset(korean_data, english_data, small_mode)
        
        return korean_data, english_data
    
    def _create_mixed_dataset(self, korean_data, english_data, small_mode):
        """혼합 데이터셋 생성"""
        logger.info("🔀 혼합 사전훈련 데이터셋 생성 중...")
        
        mixing_ratio = self.config.get('mixing_ratio', {'korean': 0.3, 'english': 0.7})
        korean_ratio = mixing_ratio.get('korean', 0.3)
        english_ratio = mixing_ratio.get('english', 0.7)
        
        mixed_data = []
        
        # 한국어 데이터 추가
        if korean_data:
            korean_count = int(len(korean_data) * korean_ratio / (korean_ratio + english_ratio))
            mixed_data.extend(korean_data[:korean_count])
            logger.info(f"   한국어: {korean_count:,}개 문서 ({korean_ratio*100:.1f}%)")
        
        # 영어 데이터 추가
        if english_data:
            english_count = int(len(english_data) * english_ratio / (korean_ratio + english_ratio))
            mixed_data.extend(english_data[:english_count])
            logger.info(f"   영어: {english_count:,}개 문서 ({english_ratio*100:.1f}%)")
        
        # 데이터 섞기
        import random
        random.shuffle(mixed_data)
        
        # 저장
        suffix = "_small" if small_mode else ""
        mixed_output_file = self.output_dir / f"mixed_pretraining_corpus{suffix}.json"
        
        with open(mixed_output_file, 'w', encoding='utf-8') as f:
            json.dump(mixed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"🎯 혼합 데이터셋 저장: {mixed_output_file}")
        logger.info(f"   총 문서 수: {len(mixed_data):,}개")


def main():
    parser = argparse.ArgumentParser(description="사전훈련용 데이터셋 다운로드 (configs/dataset 기반)")
    parser.add_argument("--config", type=str, help="설정 파일 경로 (기본: configs/dataset/pretraining.json)")
    parser.add_argument("--info", action="store_true", help="데이터셋 정보만 표시")
    parser.add_argument("--small", action="store_true", help="소량 샘플만 다운로드")
    parser.add_argument("--force", action="store_true", help="기존 데이터 무시하고 새로 다운로드")
    parser.add_argument("--korean_only", action="store_true", help="한국어만 다운로드")
    parser.add_argument("--english_only", action="store_true", help="영어만 다운로드")
    parser.add_argument("--unique_names", action="store_true", help="고유한 파일명으로 저장")
    
    args = parser.parse_args()
    
    try:
        downloader = PretrainingDatasetDownloader(args.config)
        
        if args.info:
            downloader.get_dataset_info()
            return 0
        
        # 언어별 다운로드 옵션 처리
        if args.korean_only:
            # 한국어만 다운로드하도록 설정 임시 수정
            downloader.config = {k: v for k, v in downloader.config.items() if k != 'english_config'}
        elif args.english_only:
            # 영어만 다운로드하도록 설정 임시 수정
            downloader.config = {k: v for k, v in downloader.config.items() if k != 'korean_config'}
        
        korean_data, english_data = downloader.download_datasets(
            small_mode=args.small,
            force=args.force
        )
        
        logger.info("\n🎉 사전훈련 데이터셋 다운로드 완료!")
        logger.info(f"📁 저장 위치: {downloader.output_dir}")
        if korean_data:
            logger.info(f"🇰🇷 한국어: {len(korean_data):,}개 문서")
        if english_data:
            logger.info(f"🇺🇸 영어: {len(english_data):,}개 문서")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 