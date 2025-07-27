#!/usr/bin/env python3
"""
Finetuning Dataset Download Script
파인튜닝용 데이터셋 다운로드 스크립트 (configs/dataset/finetuning.json 기반)
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# 현재 스크립트 디렉토리
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinetuningDatasetDownloader:
    """파인튜닝용 데이터셋 다운로더 (configs/dataset 기반)"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = script_dir.parent.parent / "configs" / "dataset" / "finetuning.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 프로젝트 루트 계산
        self.project_root = script_dir.parent.parent.parent.parent
        self.output_dir = self.project_root / "datasets"
        
        logger.info(f"📋 설정 파일: {self.config_path}")
        logger.info(f"📁 출력 디렉토리: {self.output_dir}")
        
    def _load_config(self):
        """설정 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _resolve_dataset_path(self, relative_path: str) -> Path:
        """상대 경로를 절대 경로로 변환"""
        if relative_path.startswith("/"):
            return Path(relative_path)
        
        # configs/dataset/ 기준으로 상대 경로 해석
        base_dir = self.config_path.parent
        return (base_dir / relative_path).resolve()
    
    def get_dataset_info(self):
        """데이터셋 정보 표시"""
        logger.info(f"📊 파인튜닝 데이터셋 설정:")
        logger.info(f"   설명: {self.config.get('description', 'N/A')}")
        logger.info(f"   태스크 유형: {self.config.get('task_type', 'N/A')}")
        
        datasets = self.config.get('datasets', [])
        if datasets:
            logger.info(f"\n📚 데이터셋 목록:")
            for i, ds in enumerate(datasets, 1):
                name = ds.get('name', 'Unknown')
                format_type = ds.get('format', 'Unknown')
                weight = ds.get('weight', 1.0)
                path = ds.get('path', 'N/A')
                logger.info(f"   {i}. {name}")
                logger.info(f"      📁 경로: {path}")
                logger.info(f"      🎯 형식: {format_type}")
                logger.info(f"      ⚖️  가중치: {weight}")
        
        # 전처리 설정
        preprocessing = self.config.get('preprocessing', {})
        if preprocessing:
            logger.info(f"\n🔧 전처리 설정:")
            max_len = preprocessing.get('max_sequence_length', 'N/A')
            template = preprocessing.get('prompt_template', 'N/A')
            logger.info(f"   최대 시퀀스 길이: {max_len}")
            logger.info(f"   프롬프트 템플릿: {template[:50]}..." if len(str(template)) > 50 else f"   프롬프트 템플릿: {template}")
    
    def validate_datasets(self) -> bool:
        """데이터셋 파일 존재 여부 확인"""
        logger.info("🔍 데이터셋 파일 유효성 검사...")
        
        datasets = self.config.get('datasets', [])
        all_valid = True
        
        for ds in datasets:
            name = ds.get('name', 'Unknown')
            path = ds.get('path', '')
            
            # 경로 해석
            if path.startswith('../../../../datasets/'):
                # 프로젝트 루트의 datasets 디렉토리 참조
                dataset_path = self.project_root / "datasets" / Path(path).name
            else:
                dataset_path = self._resolve_dataset_path(path)
            
            if dataset_path.exists():
                file_size = dataset_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"   ✅ {name}: {dataset_path} ({file_size:.1f} MB)")
            else:
                logger.warning(f"   ❌ {name}: {dataset_path} (파일 없음)")
                all_valid = False
        
        return all_valid
    
    def create_sample_datasets(self):
        """샘플 파인튜닝 데이터셋 생성"""
        logger.info("📝 샘플 파인튜닝 데이터셋 생성 중...")
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 한국어 명령어 데이터 샘플
        korean_instructions = [
            {
                "instruction": "다음 텍스트를 요약해주세요.",
                "input": "인공지능은 인간의 지능을 모방하여 학습, 추론, 판단 등의 기능을 수행하는 기술입니다. 최근 딥러닝과 머신러닝의 발전으로 다양한 분야에서 활용되고 있습니다.",
                "output": "인공지능은 인간의 지능을 모방한 기술로, 딥러닝과 머신러닝 발전에 따라 다양한 분야에서 활용되고 있습니다."
            },
            {
                "instruction": "파이썬으로 간단한 계산기 함수를 만들어주세요.",
                "input": "덧셈, 뺄셈, 곱셈, 나눗셈이 가능한 계산기",
                "output": "```python\ndef calculator(a, b, operation):\n    if operation == '+':\n        return a + b\n    elif operation == '-':\n        return a - b\n    elif operation == '*':\n        return a * b\n    elif operation == '/':\n        return a / b if b != 0 else '0으로 나눌 수 없습니다'\n    else:\n        return '지원하지 않는 연산입니다'\n```"
            },
            {
                "instruction": "다음 문장을 영어로 번역해주세요.",
                "input": "오늘 날씨가 정말 좋네요.",
                "output": "The weather is really nice today."
            }
        ]
        
        # 영어 명령어 데이터 샘플
        english_instructions = [
            {
                "instruction": "Explain what machine learning is in simple terms.",
                "input": "",
                "output": "Machine learning is a type of artificial intelligence that allows computers to learn and make decisions from data without being explicitly programmed for every task."
            },
            {
                "instruction": "Write a Python function to check if a number is prime.",
                "input": "",
                "output": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"
            },
            {
                "instruction": "Translate the following sentence to Korean.",
                "input": "I love learning new languages.",
                "output": "저는 새로운 언어를 배우는 것을 좋아합니다."
            }
        ]
        
        # 파일 저장
        korean_file = self.output_dir / "korean_instructions.json"
        english_file = self.output_dir / "english_instructions.json"
        
        with open(korean_file, 'w', encoding='utf-8') as f:
            json.dump(korean_instructions, f, ensure_ascii=False, indent=2)
        
        with open(english_file, 'w', encoding='utf-8') as f:
            json.dump(english_instructions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 한국어 명령어 데이터: {korean_file} ({len(korean_instructions)}개)")
        logger.info(f"✅ 영어 명령어 데이터: {english_file} ({len(english_instructions)}개)")
        
        return korean_file, english_file
    
    def process_datasets(self):
        """파인튜닝 데이터셋 처리"""
        logger.info("🔄 파인튜닝 데이터셋 처리 중...")
        
        datasets = self.config.get('datasets', [])
        preprocessing = self.config.get('preprocessing', {})
        
        all_data = []
        
        for ds in datasets:
            name = ds.get('name', 'Unknown')
            path = ds.get('path', '')
            format_type = ds.get('format', 'alpaca')
            weight = ds.get('weight', 1.0)
            
            # 경로 해석
            if path.startswith('../../../../datasets/'):
                dataset_path = self.project_root / "datasets" / Path(path).name
            else:
                dataset_path = self._resolve_dataset_path(path)
            
            if not dataset_path.exists():
                logger.warning(f"⚠️  {name} 파일이 없습니다: {dataset_path}")
                continue
            
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 가중치 적용 (데이터 복제)
                weighted_data = data * int(weight)
                all_data.extend(weighted_data)
                
                logger.info(f"✅ {name}: {len(data)}개 → {len(weighted_data)}개 (가중치 {weight})")
                
            except Exception as e:
                logger.error(f"❌ {name} 로드 실패: {e}")
        
        # 처리된 데이터 저장
        if all_data:
            output_file = self.output_dir / "processed_finetuning_data.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🎯 처리된 파인튜닝 데이터: {output_file}")
            logger.info(f"   총 예시 수: {len(all_data):,}개")
            
            return output_file
        
        return None


def main():
    parser = argparse.ArgumentParser(description="파인튜닝용 데이터셋 다운로드 (configs/dataset 기반)")
    parser.add_argument("--config", type=str, help="설정 파일 경로 (기본: configs/dataset/finetuning.json)")
    parser.add_argument("--info", action="store_true", help="데이터셋 정보만 표시")
    parser.add_argument("--validate", action="store_true", help="데이터셋 파일 유효성 검사")
    parser.add_argument("--create_samples", action="store_true", help="샘플 데이터셋 생성")
    parser.add_argument("--process", action="store_true", help="데이터셋 처리 및 병합")
    
    args = parser.parse_args()
    
    try:
        downloader = FinetuningDatasetDownloader(args.config)
        
        if args.info:
            downloader.get_dataset_info()
            return 0
        
        if args.validate:
            is_valid = downloader.validate_datasets()
            if is_valid:
                logger.info("✅ 모든 데이터셋 파일이 유효합니다.")
            else:
                logger.warning("⚠️  일부 데이터셋 파일에 문제가 있습니다.")
            return 0 if is_valid else 1
        
        if args.create_samples:
            korean_file, english_file = downloader.create_sample_datasets()
            logger.info(f"\n🎉 샘플 데이터셋 생성 완료!")
            logger.info(f"📁 저장 위치: {downloader.output_dir}")
            return 0
        
        if args.process:
            output_file = downloader.process_datasets()
            if output_file:
                logger.info(f"\n🎉 파인튜닝 데이터셋 처리 완료!")
                logger.info(f"📁 출력 파일: {output_file}")
            else:
                logger.warning("처리할 데이터가 없습니다.")
            return 0
        
        # 기본 동작: 정보 표시
        downloader.get_dataset_info()
        return 0
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 