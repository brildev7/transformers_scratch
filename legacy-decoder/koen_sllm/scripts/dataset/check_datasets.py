#!/usr/bin/env python3
"""
Independent dataset checker script
독립적인 다운로드된 데이터셋 확인 스크립트
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 현재 스크립트 디렉토리 
script_dir = Path(__file__).parent


def check_dataset_file(file_path: Path, dataset_name: str) -> Optional[Dict]:
    """데이터셋 파일 확인"""
    if not file_path.exists():
        logger.warning(f"❌ {dataset_name} 파일이 없습니다: {file_path}")
        return None
    
    try:
        # 파일 크기
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        # 내용 확인
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error(f"❌ {dataset_name} 파일 형식이 잘못되었습니다. 리스트 형태여야 합니다.")
            return None
        
        num_docs = len(data)
        
        # 샘플 텍스트 길이 확인
        if num_docs > 0:
            sample_lengths = []
            valid_texts = 0
            
            for i, text in enumerate(data[:min(100, num_docs)]):
                if isinstance(text, str) and text.strip():
                    sample_lengths.append(len(text))
                    valid_texts += 1
            
            if sample_lengths:
                avg_length = sum(sample_lengths) / len(sample_lengths)
                min_length = min(sample_lengths)
                max_length = max(sample_lengths)
            else:
                avg_length = min_length = max_length = 0
                
            # 유효한 텍스트 비율
            valid_ratio = valid_texts / min(100, num_docs) * 100
        else:
            avg_length = min_length = max_length = 0
            valid_ratio = 0
        
        info = {
            'file_size_mb': size_mb,
            'num_documents': num_docs,
            'avg_text_length': avg_length,
            'min_text_length': min_length,
            'max_text_length': max_length,
            'valid_text_ratio': valid_ratio,
            'sample_texts': data[:3] if num_docs > 0 else []
        }
        
        logger.info(f"✅ {dataset_name} 데이터셋:")
        logger.info(f"   📁 파일 크기: {size_mb:.1f} MB")
        logger.info(f"   📄 문서 수: {num_docs:,}개")
        logger.info(f"   📏 평균 텍스트 길이: {avg_length:.0f}자")
        logger.info(f"   📏 텍스트 길이 범위: {min_length}~{max_length}자")
        logger.info(f"   ✔️  유효 텍스트 비율: {valid_ratio:.1f}%")
        
        return info
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ {dataset_name} JSON 파싱 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ {dataset_name} 파일 읽기 실패: {e}")
        return None


def show_sample_texts(korean_data: Optional[Dict], english_data: Optional[Dict], num_samples: int = 3):
    """샘플 텍스트 표시"""
    logger.info("📝 샘플 텍스트:")
    
    if korean_data and korean_data['sample_texts']:
        logger.info("\n🇰🇷 한국어 샘플:")
        for i, text in enumerate(korean_data['sample_texts'][:num_samples]):
            if isinstance(text, str):
                preview = text[:200] + "..." if len(text) > 200 else text
                logger.info(f"   {i+1}. {preview}")
    
    if english_data and english_data['sample_texts']:
        logger.info("\n🇺🇸 영어 샘플:")
        for i, text in enumerate(english_data['sample_texts'][:num_samples]):
            if isinstance(text, str):
                preview = text[:200] + "..." if len(text) > 200 else text
                logger.info(f"   {i+1}. {preview}")


def check_disk_usage(data_dir: Path):
    """디스크 사용량 확인"""
    total_size = 0
    file_count = 0
    
    if data_dir.exists():
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
    
    total_mb = total_size / (1024 * 1024)
    total_gb = total_mb / 1024
    
    logger.info(f"💾 데이터 디렉토리 사용량:")
    logger.info(f"   📁 총 파일 수: {file_count}개")
    logger.info(f"   📦 총 크기: {total_mb:.1f} MB ({total_gb:.2f} GB)")


def analyze_data_quality(korean_data: Optional[Dict], english_data: Optional[Dict]) -> List[str]:
    """데이터 품질 분석"""
    quality_issues = []
    
    # 한국어 데이터 체크
    if korean_data:
        if korean_data['avg_text_length'] < 50:
            quality_issues.append("⚠️  한국어 텍스트가 너무 짧습니다 (평균 50자 미만)")
        if korean_data['num_documents'] < 100:
            quality_issues.append("⚠️  한국어 문서 수가 부족합니다 (100개 미만)")
        if korean_data['valid_text_ratio'] < 80:
            quality_issues.append(f"⚠️  한국어 유효 텍스트 비율이 낮습니다 ({korean_data['valid_text_ratio']:.1f}%)")
    else:
        quality_issues.append("❌ 한국어 데이터가 없습니다")
    
    # 영어 데이터 체크
    if english_data:
        if english_data['avg_text_length'] < 50:
            quality_issues.append("⚠️  영어 텍스트가 너무 짧습니다 (평균 50자 미만)")
        if english_data['num_documents'] < 100:
            quality_issues.append("⚠️  영어 문서 수가 부족합니다 (100개 미만)")
        if english_data['valid_text_ratio'] < 80:
            quality_issues.append(f"⚠️  영어 유효 텍스트 비율이 낮습니다 ({english_data['valid_text_ratio']:.1f}%)")
    else:
        quality_issues.append("❌ 영어 데이터가 없습니다")
    
    return quality_issues


def provide_recommendations(total_docs: int, quality_issues: List[str]):
    """사용자에게 권장사항 제공"""
    logger.info("\n💡 권장사항:")
    
    has_critical_issues = any("❌" in issue for issue in quality_issues)
    
    if has_critical_issues:
        logger.info("   🔄 데이터가 없거나 부족합니다. 다운로드를 먼저 실행하세요:")
        logger.info("      python3 common/scripts/download_all.py --small")
    elif total_docs < 1000:
        logger.info("   🔄 더 많은 데이터를 다운로드하는 것을 권장합니다:")
        logger.info("      python3 common/scripts/download_all.py")
    elif total_docs < 10000:
        logger.info("   📊 테스트용으로는 충분하지만, 실제 학습을 위해서는 더 많은 데이터가 필요합니다")
        logger.info("      python3 common/scripts/download_all.py")
    else:
        logger.info("   ✅ 학습에 충분한 데이터가 있습니다")
        logger.info("   🚀 이제 원하는 모델에서 데이터를 사용할 수 있습니다")
    
    logger.info("\n🔧 추가 도구:")
    logger.info("   • 셸 스크립트 사용: ./common/scripts/download_datasets.sh")
    logger.info("   • 영어만: python3 common/scripts/download_english.py")
    logger.info("   • 한국어만: python3 common/scripts/download_korean.py")


def main():
    parser = argparse.ArgumentParser(description="독립적인 데이터셋 확인 도구")
    parser.add_argument("--data_dir", type=str, default="../../../../datasets", help="데이터 디렉토리")
    parser.add_argument("--show_samples", action="store_true", help="샘플 텍스트 표시")
    parser.add_argument("--export_stats", type=str, help="통계를 JSON 파일로 저장")
    parser.add_argument("--verbose", action="store_true", help="상세한 출력")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    data_dir = Path(args.data_dir)
    korean_path = data_dir / "korean_corpus.json"
    english_path = data_dir / "english_corpus.json"
    
    logger.info("=" * 60)
    logger.info("📊 독립적인 데이터셋 현황 확인")
    logger.info(f"📁 데이터 디렉토리: {data_dir}")
    logger.info("=" * 60)
    
    # 데이터셋 파일 확인
    korean_data = check_dataset_file(korean_path, "한국어")
    english_data = check_dataset_file(english_path, "영어")
    
    # 전체 통계
    total_docs = 0
    total_size = 0
    
    if korean_data:
        total_docs += korean_data['num_documents']
        total_size += korean_data['file_size_mb']
    
    if english_data:
        total_docs += english_data['num_documents']
        total_size += english_data['file_size_mb']
    
    logger.info("\n📈 전체 통계:")
    logger.info(f"   📄 총 문서 수: {total_docs:,}개")
    logger.info(f"   💾 총 데이터 크기: {total_size:.1f} MB")
    
    # 디스크 사용량
    logger.info("")
    check_disk_usage(data_dir)
    
    # 샘플 텍스트 표시
    if args.show_samples:
        logger.info("")
        show_sample_texts(korean_data, english_data)
    
    # 데이터 품질 체크
    logger.info("\n🔍 데이터 품질 체크:")
    quality_issues = analyze_data_quality(korean_data, english_data)
    
    if quality_issues:
        for issue in quality_issues:
            logger.warning(f"   {issue}")
    else:
        logger.info("   ✅ 데이터 품질이 양호합니다")
    
    # 권장사항 제공
    provide_recommendations(total_docs, quality_issues)
    
    # 통계 내보내기
    if args.export_stats:
        from datetime import datetime
        
        stats = {
            'korean_data': korean_data,
            'english_data': english_data,
            'total_documents': total_docs,
            'total_size_mb': total_size,
            'quality_issues': quality_issues,
            'check_timestamp': datetime.now().isoformat(),
            'data_directory': str(data_dir)
        }
        
        export_path = Path(args.export_stats)
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"\n📊 통계가 {export_path}에 저장되었습니다")
    
    logger.info("=" * 60)
    
    # 종료 코드 결정
    has_critical_issues = any("❌" in issue for issue in quality_issues)
    return 1 if has_critical_issues else 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 