#!/usr/bin/env python3
"""
Dynamic Dataset Loader
JSON 설정 기반 동적 데이터셋 로더
"""
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


class DatasetConfig:
    """데이터셋 설정 클래스"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.name = config_dict['name']
        self.config = config_dict.get('config')
        self.split = config_dict['split']
        self.description = config_dict['description']
        self.text_field = config_dict['text_field']
        self.limits = config_dict['limits']
        self.min_text_length = config_dict['min_text_length']
        self.max_text_length = config_dict.get('max_text_length')
        self.streaming = config_dict.get('streaming', False)
        self.enabled = config_dict.get('enabled', True)
        self.priority = config_dict.get('priority', 999)
        self.fallback = config_dict.get('fallback', {})
        self.filter = config_dict.get('filter', {})
        self.note = config_dict.get('note', '')
    
    def get_limit(self, small_mode: bool) -> int:
        """모드에 따른 문서 제한 수 반환"""
        return self.limits['small'] if small_mode else self.limits['full']
    
    def __repr__(self):
        return f"DatasetConfig(name='{self.name}', description='{self.description}')"


class DynamicDatasetLoader:
    """JSON 설정 기반 동적 데이터셋 로더"""
    
    def __init__(self, config_file_path: str, cache_dir: str):
        self.config_file_path = Path(config_file_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 중복 제거를 위한 해시 세트
        self.text_hashes: Set[str] = set()
        
        # JSON 설정 로드
        self.config = self._load_config()
        self.global_settings = self.config.get('global_settings', {})
        self.datasets = self._parse_datasets()
        
        logger.info(f"설정 파일 로드: {self.config_file_path}")
        logger.info(f"활성화된 데이터셋: {len(self.datasets)}개")
    
    def _load_config(self) -> Dict[str, Any]:
        """JSON 설정 파일 로드"""
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {self.config_file_path}")
            logger.error(f"오류: {e}")
            raise
    
    def _parse_datasets(self) -> List[DatasetConfig]:
        """데이터셋 설정 파싱"""
        datasets = []
        for dataset_dict in self.config.get('datasets', []):
            if dataset_dict.get('enabled', True):
                datasets.append(DatasetConfig(dataset_dict))
        
        # 우선순위로 정렬
        datasets.sort(key=lambda x: x.priority)
        return datasets
    
    def _get_text_hash(self, text: str) -> str:
        """텍스트의 해시값 생성"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, text: str) -> bool:
        """중복 텍스트인지 확인"""
        text_hash = self._get_text_hash(text)
        if text_hash in self.text_hashes:
            return True
        self.text_hashes.add(text_hash)
        return False
    
    def load_dataset_safely(self, dataset_config: DatasetConfig, small_mode: bool = False) -> List[str]:
        """안전하게 데이터셋 로드"""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets 라이브러리가 필요합니다: pip install datasets")
            return []
        
        logger.info(f"{dataset_config.description} 로드 중...")
        texts = []
        
        try:
            # 데이터셋 로드 파라미터 구성
            load_params = {
                'path': dataset_config.name,
                'split': dataset_config.split,
                'cache_dir': str(self.cache_dir)
            }
            
            if dataset_config.config:
                load_params['name'] = dataset_config.config
            
            if dataset_config.streaming:
                load_params['streaming'] = True
            
            # 데이터셋 로드
            dataset = load_dataset(**load_params)
            
            # 제한 수 설정
            limit = dataset_config.get_limit(small_mode)
            
            # 스트리밍 vs 일반 처리
            if dataset_config.streaming:
                texts = self._process_streaming_dataset(dataset, dataset_config, limit)
            else:
                texts = self._process_regular_dataset(dataset, dataset_config, limit)
            
            logger.info(f"{dataset_config.description}: {len(texts)}개 문서")
            return texts
            
        except Exception as e:
            logger.warning(f"{dataset_config.description} 로드 실패: {e}")
            
            # fallback 처리
            fallback_action = dataset_config.fallback.get('action', 'skip')
            if fallback_action == 'skip':
                logger.info(f"{dataset_config.description} 스킵됨")
            elif fallback_action == 'continue':
                logger.info(f"{dataset_config.description} 실패했지만 계속 진행")
            
            return []
    
    def _process_streaming_dataset(self, dataset, config: DatasetConfig, limit: int) -> List[str]:
        """스트리밍 데이터셋 처리"""
        texts = []
        for i, example in enumerate(dataset):
            if i >= limit:
                break
            
            text = self._extract_text(example, config)
            if text and self._validate_text(text, config) and not self._is_duplicate(text):
                texts.append(text)
        
        return texts
    
    def _process_regular_dataset(self, dataset, config: DatasetConfig, limit: int) -> List[str]:
        """일반 데이터셋 처리"""
        texts = []
        
        # 데이터셋 크기 제한
        if len(dataset) > limit:
            dataset = dataset.select(range(limit))
        
        for example in dataset:
            text = self._extract_text(example, config)
            if text and self._validate_text(text, config) and not self._is_duplicate(text):
                texts.append(text)
        
        return texts
    
    def _extract_text(self, example: Dict, config: DatasetConfig) -> Optional[str]:
        """예제에서 텍스트 추출"""
        text = example.get(config.text_field, '').strip()
        
        # 언어 필터 적용
        if config.filter:
            lang_field = config.filter.get('language_field')
            lang_value = config.filter.get('language_value')
            if lang_field and lang_value:
                if lang_value.lower() not in example.get(lang_field, '').lower():
                    return None
        
        # 길이 제한 적용
        if config.max_text_length and len(text) > config.max_text_length:
            text = text[:config.max_text_length]
        
        return text
    
    def _validate_text(self, text: str, config: DatasetConfig) -> bool:
        """텍스트 유효성 검증"""
        if len(text) < config.min_text_length:
            return False
        
        return True
    
    def load_all_datasets(self, small_mode: bool = False) -> List[str]:
        """모든 활성화된 데이터셋 로드"""
        all_texts = []
        total_processed = 0
        duplicates_removed = 0
        
        logger.info("🔄 중복 제거 기능 활성화")
        
        for dataset_config in self.datasets:
            initial_hash_count = len(self.text_hashes)
            texts = self.load_dataset_safely(dataset_config, small_mode)
            
            # 중복 제거 통계 계산
            final_hash_count = len(self.text_hashes)
            dataset_duplicates = (initial_hash_count + len(texts)) - final_hash_count
            
            total_processed += len(texts) + dataset_duplicates
            duplicates_removed += dataset_duplicates
            
            if dataset_duplicates > 0:
                logger.info(f"   ♻️  중복 제거: {dataset_duplicates}개")
            
            all_texts.extend(texts)
        
        # 최종 통계
        logger.info(f"\n📊 데이터 로드 완료:")
        logger.info(f"   총 처리: {total_processed:,}개 문서")
        logger.info(f"   중복 제거: {duplicates_removed:,}개 문서")
        logger.info(f"   최종 데이터: {len(all_texts):,}개 문서")
        logger.info(f"   중복률: {(duplicates_removed/total_processed*100):.1f}%" if total_processed > 0 else "   중복률: 0.0%")
        
        return all_texts
    
    def save_texts(self, texts: List[str], output_path: Optional[str] = None) -> Path:
        """텍스트를 JSON 파일로 저장"""
        if output_path is None:
            output_path = self.global_settings.get('output_filename', 'corpus.json')
        
        output_file = Path(output_path)
        
        # 전역 설정에서 JSON 저장 옵션 가져오기
        indent = self.global_settings.get('json_indent', 2)
        encoding = self.global_settings.get('encoding', 'utf-8')
        
        with open(output_file, 'w', encoding=encoding) as f:
            json.dump(texts, f, ensure_ascii=False, indent=indent)
        
        logger.info(f"데이터 저장 완료: {output_file} ({len(texts):,}개 문서)")
        return output_file
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        enabled_datasets = [
            {
                'name': ds.name,
                'description': ds.description,
                'priority': ds.priority,
                'enabled': ds.enabled
            }
            for ds in self.datasets
        ]
        
        return {
            'config_file': str(self.config_file_path),
            'total_datasets': len(self.datasets),
            'enabled_datasets': enabled_datasets,
            'global_settings': self.global_settings
        } 