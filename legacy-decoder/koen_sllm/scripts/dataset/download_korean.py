#!/usr/bin/env python3
"""
Dynamic Korean dataset download script
JSON 설정 기반 한국어 데이터셋 다운로드 스크립트
"""
import sys
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


def main():
    parser = argparse.ArgumentParser(description="JSON 설정 기반 한국어 데이터셋 다운로드")
    parser.add_argument("--force", action="store_true", help="기존 데이터 무시하고 새로 다운로드")
    parser.add_argument("--small", action="store_true", help="소량 샘플만 다운로드")
    parser.add_argument("--output_dir", type=str, default="../../../../datasets", help="출력 디렉토리")
    parser.add_argument("--config", type=str, 
                       default=str(script_dir.parent.parent / "configs" / "dataset" / "korean_datasets.json"),
                       help="한국어 데이터셋 설정 파일")
    parser.add_argument("--info", action="store_true", help="데이터셋 설정 정보만 표시")
    
    args = parser.parse_args()
    
    try:
        # 설정 파일 확인
        config_file = Path(args.config)
        if not config_file.exists():
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_file}")
            logger.error("다음 명령으로 기본 설정을 확인하세요:")
            logger.error(f"ls -la {script_dir.parent.parent / 'configs' / 'training'}/")
            return 1
        
        # 캐시 디렉토리 설정
        cache_dir = script_dir.parent.parent.parent.parent / "models"
        
        # 동적 로더 생성
        loader = DynamicDatasetLoader(str(config_file), str(cache_dir))
        
        # 정보만 표시하는 모드
        if args.info:
            logger.info("📋 한국어 데이터셋 설정 정보:")
            info = loader.get_dataset_info()
            
            logger.info(f"설정 파일: {info['config_file']}")
            logger.info(f"활성화된 데이터셋: {info['total_datasets']}개")
            
            logger.info("\n📊 데이터셋 목록:")
            for i, ds in enumerate(info['enabled_datasets'], 1):
                logger.info(f"   {i}. {ds['description']} (우선순위: {ds['priority']})")
            
            return 0
        
        logger.info("한국어 데이터셋 다운로드 시작...")
        
        # 출력 파일 경로 설정
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = loader.global_settings.get('output_filename', 'korean_corpus.json')
        korean_path = output_dir / output_filename
        
        # 기존 파일 체크
        if korean_path.exists() and not args.force:
            logger.info(f"기존 한국어 데이터가 있습니다: {korean_path}")
            logger.info("새로 다운로드하려면 --force 옵션을 사용하세요.")
            logger.info("또는 --info 옵션으로 설정 정보를 확인하세요.")
            return 0
        
        if args.force and korean_path.exists():
            logger.info("기존 데이터를 덮어쓰겠습니다...")
            korean_path.unlink()
        
        # 데이터셋 정보 표시
        info = loader.get_dataset_info()
        logger.info(f"📋 로드할 데이터셋: {info['total_datasets']}개")
        for ds in info['enabled_datasets']:
            logger.info(f"   • {ds['description']}")
        
        # 모든 데이터셋 다운로드
        korean_texts = loader.load_all_datasets(small_mode=args.small)
        
        if not korean_texts:
            logger.warning("다운로드된 텍스트가 없습니다.")
            logger.info("데이터셋 설정을 확인하거나 네트워크 연결을 확인하세요.")
            return 1
        
        # 데이터 저장
        saved_file = loader.save_texts(korean_texts, str(korean_path))
        
        logger.info(f"✅ 한국어 데이터셋 다운로드 완료!")
        logger.info(f"   - 문서 수: {len(korean_texts):,}개")
        logger.info(f"   - 저장 위치: {saved_file}")
        
        # 파일 크기 표시
        if saved_file.exists():
            file_size = saved_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"   - 파일 크기: {file_size:.1f} MB")
        
        # 설정 정보 표시
        logger.info(f"\n🔧 사용된 설정 파일: {config_file}")
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