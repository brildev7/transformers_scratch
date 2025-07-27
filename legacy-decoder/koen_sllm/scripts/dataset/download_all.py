#!/usr/bin/env python3
"""
Dynamic all datasets download script
JSON 설정 기반 전체 데이터셋 다운로드 스크립트
"""
import sys
import argparse
import time
import logging
from pathlib import Path

# 현재 스크립트 디렉토리
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from dataset_loader import DynamicDatasetLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="JSON 설정 기반 전체 데이터셋 다운로드")
    parser.add_argument("--force", action="store_true", help="기존 데이터 무시하고 새로 다운로드")
    parser.add_argument("--small", action="store_true", help="소량 샘플만 다운로드")
    parser.add_argument("--output_dir", type=str, default="../../../../datasets", help="출력 디렉토리")
    parser.add_argument("--korean_only", action="store_true", help="한국어만 다운로드")
    parser.add_argument("--english_only", action="store_true", help="영어만 다운로드")
    parser.add_argument("--korean_config", type=str, 
                       default=str(script_dir.parent.parent / "configs" / "dataset" / "korean_datasets.json"),
                       help="한국어 데이터셋 설정 파일")
    parser.add_argument("--english_config", type=str, 
                       default=str(script_dir.parent.parent / "configs" / "dataset" / "english_datasets.json"),
                       help="영어 데이터셋 설정 파일")
    parser.add_argument("--info", action="store_true", help="데이터셋 설정 정보만 표시")
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        
        # 설정 파일 경로들
        korean_config = Path(args.korean_config)
        english_config = Path(args.english_config)
        
        # 정보 표시 모드
        if args.info:
            logger.info("📋 전체 데이터셋 설정 정보:")
            
            # 한국어 설정 정보
            if korean_config.exists() and not args.english_only:
                logger.info("\n🇰🇷 한국어 데이터셋:")
                korean_loader = DynamicDatasetLoader(str(korean_config), str(script_dir.parent.parent.parent.parent / "models"))
                korean_info = korean_loader.get_dataset_info()
                for i, ds in enumerate(korean_info['enabled_datasets'], 1):
                    logger.info(f"   {i}. {ds['description']}")
            
            # 영어 설정 정보
            if english_config.exists() and not args.korean_only:
                logger.info("\n🇺🇸 영어 데이터셋:")
                english_loader = DynamicDatasetLoader(str(english_config), str(script_dir.parent.parent.parent.parent / "models"))
                english_info = english_loader.get_dataset_info()
                for i, ds in enumerate(english_info['enabled_datasets'], 1):
                    logger.info(f"   {i}. {ds['description']}")
            
            return 0
        
        logger.info("전체 데이터셋 다운로드 시작...")
        
        # 출력 디렉토리 생성
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        korean_data = []
        english_data = []
        
        # 한국어 데이터 다운로드
        if not args.english_only and korean_config.exists():
            logger.info("🇰🇷 한국어 데이터셋 다운로드 중...")
            
            korean_loader = DynamicDatasetLoader(str(korean_config), str(script_dir.parent.parent.parent.parent / "models"))
            korean_output_file = output_dir / korean_loader.global_settings.get('output_filename', 'korean_corpus.json')
            
            # 기존 파일 체크
            if korean_output_file.exists() and not args.force:
                logger.info(f"기존 한국어 데이터가 있습니다: {korean_output_file}")
            else:
                if args.force and korean_output_file.exists():
                    korean_output_file.unlink()
                    logger.info("기존 한국어 데이터 삭제")
                
                korean_data = korean_loader.load_all_datasets(small_mode=args.small)
                if korean_data:
                    korean_loader.save_texts(korean_data, str(korean_output_file))
        
        # 영어 데이터 다운로드
        if not args.korean_only and english_config.exists():
            logger.info("🇺🇸 영어 데이터셋 다운로드 중...")
            
            english_loader = DynamicDatasetLoader(str(english_config), str(script_dir.parent.parent.parent.parent / "models"))
            english_output_file = output_dir / english_loader.global_settings.get('output_filename', 'english_corpus.json')
            
            # 기존 파일 체크
            if english_output_file.exists() and not args.force:
                logger.info(f"기존 영어 데이터가 있습니다: {english_output_file}")
            else:
                if args.force and english_output_file.exists():
                    english_output_file.unlink()
                    logger.info("기존 영어 데이터 삭제")
                
                english_data = english_loader.load_all_datasets(small_mode=args.small)
                if english_data:
                    english_loader.save_texts(english_data, str(english_output_file))
        
        # 결과 요약
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("✅ 데이터셋 다운로드 완료!")
        logger.info(f"⏱️  소요 시간: {total_time/60:.1f}분")
        
        korean_path = output_dir / "korean_corpus.json"
        english_path = output_dir / "english_corpus.json"
        
        if korean_path.exists():
            korean_size = korean_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"🇰🇷 한국어 데이터: {korean_size:.1f} MB")
        
        if english_path.exists():
            english_size = english_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"🇺🇸 영어 데이터: {english_size:.1f} MB")
        
        total_docs = len(korean_data) + len(english_data)
        logger.info(f"📊 총 문서: {total_docs:,}개")
        
        # 총 디스크 사용량
        total_size = 0
        if korean_path.exists():
            total_size += korean_path.stat().st_size
        if english_path.exists():
            total_size += english_path.stat().st_size
        
        total_size_mb = total_size / (1024 * 1024)
        logger.info(f"💾 총 크기: {total_size_mb:.1f} MB")
        
        logger.info("=" * 60)
        logger.info("🚀 다음 단계:")
        logger.info("1. python3 common/scripts/check_datasets.py  # 데이터 확인")
        logger.info("2. 원하는 모델에서 데이터를 활용하여 학습 시작")
        
        logger.info("\n🔧 설정 파일 관리:")
        logger.info(f"   • 한국어: {korean_config}")
        logger.info(f"   • 영어: {english_config}")
        logger.info("💡 데이터셋을 추가/제거하려면 JSON 설정 파일을 편집하세요.")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 