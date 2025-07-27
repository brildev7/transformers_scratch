#!/usr/bin/env python3
"""
한국어 sLLM 미세조정 데이터 전처리 모듈
Finetuning data preprocessing module for Korean sLLM

이 모듈은 다운로드된 원시 명령어 데이터를 미세조정에 적합하도록 전처리합니다.
최소 5만개 이상의 고품질 데이터를 생성하며, 한영 혼합을 고려합니다.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import random
from collections import defaultdict
import copy

# 프로젝트 루트 추가
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from tqdm import tqdm
import hashlib

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess_finetuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinetuningDataProcessor:
    """미세조정 데이터 전처리 클래스"""
    
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
        
        # 목표 데이터 개수 설정
        self.min_target_count = 50000  # 최소 5만개
        self.target_count = 100000     # 목표 10만개
        
        # 한영 혼합 비율 설정
        self.language_ratios = {
            "ko": 0.6,  # 한국어 60%
            "en": 0.4   # 영어 40%
        }
        
        # 품질 필터링 설정
        self.min_instruction_length = 10
        self.max_instruction_length = 500
        self.min_output_length = 10
        self.max_output_length = 2000
        
        # 중복 제거용 해시셋
        self.seen_hashes = set()
        
        # 처리 통계
        self.stats = {
            "total_processed": 0,
            "korean_instructions": 0,
            "english_instructions": 0,
            "filtered_out": 0,
            "duplicates_removed": 0,
            "augmented_data": 0
        }
        
        # 태스크 분류
        self.task_categories = {
            "qa": "질문답변",
            "summarization": "요약",
            "translation": "번역",
            "classification": "분류",
            "generation": "생성",
            "conversation": "대화",
            "reasoning": "추론",
            "math": "수학",
            "coding": "코딩",
            "general": "일반"
        }
    
    def _extract_instruction_data(self, item: Dict, source_name: str) -> Optional[Dict]:
        """
        데이터 아이템에서 명령어 데이터 추출
        
        Args:
            item: 데이터 아이템
            source_name: 데이터 소스 이름
            
        Returns:
            표준 형식의 명령어 데이터 또는 None
        """
        instruction = None
        input_text = ""
        output = None
        language = item.get('language', 'ko')
        
        try:
            # 소스별 데이터 추출 로직
            if "alpaca" in source_name:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")
                
            elif "chatgpt" in source_name or "evol" in source_name:
                if "conversations" in item:
                    conversations = item["conversations"]
                    if len(conversations) >= 2:
                        instruction = conversations[0].get("value", "")
                        output = conversations[1].get("value", "")
                else:
                    instruction = item.get("instruction", "")
                    output = item.get("output", "")
                    
            elif "sharegpt" in source_name or "ultrachat" in source_name:
                if "messages" in item:
                    messages = item["messages"]
                    if len(messages) >= 2:
                        instruction = messages[0].get("content", "")
                        output = messages[1].get("content", "")
                elif "conversations" in item:
                    conversations = item["conversations"]
                    if len(conversations) >= 2:
                        instruction = conversations[0].get("value", "")
                        output = conversations[1].get("value", "")
                        
            elif "dolly" in source_name:
                instruction = item.get("instruction", "")
                context = item.get("context", "")
                if context:
                    input_text = context
                output = item.get("response", "")
                
            elif "oasst" in source_name:
                instruction = item.get("text", "")
                if item.get("role") == "assistant":
                    output = instruction
                    instruction = item.get("parent_text", "")
                else:
                    output = item.get("response", "")
                    
            else:
                # 일반적인 필드들 시도
                instruction = (item.get("instruction", "") or 
                             item.get("question", "") or
                             item.get("input", "") or
                             item.get("prompt", ""))
                output = (item.get("output", "") or
                         item.get("answer", "") or
                         item.get("response", "") or
                         item.get("completion", ""))
                input_text = item.get("input", "") or item.get("context", "")
            
            # 데이터 검증
            if not instruction or not output:
                return None
            
            # 표준 형식으로 변환
            return {
                "instruction": instruction.strip(),
                "input": input_text.strip(),
                "output": output.strip(),
                "language": language,
                "source": source_name,
                "task_category": self._classify_task(instruction, output)
            }
            
        except Exception as e:
            logger.warning(f"데이터 추출 오류: {source_name} - {e}")
            return None
    
    def _classify_task(self, instruction: str, output: str) -> str:
        """
        태스크 분류
        
        Args:
            instruction: 명령어
            output: 응답
            
        Returns:
            태스크 카테고리
        """
        instruction_lower = instruction.lower()
        
        # 키워드 기반 분류
        if any(keyword in instruction_lower for keyword in ["what", "who", "where", "when", "why", "how", "무엇", "누구", "어디", "언제", "왜", "어떻게"]):
            return "qa"
        elif any(keyword in instruction_lower for keyword in ["summarize", "summary", "요약"]):
            return "summarization"
        elif any(keyword in instruction_lower for keyword in ["translate", "번역"]):
            return "translation"
        elif any(keyword in instruction_lower for keyword in ["classify", "category", "분류"]):
            return "classification"
        elif any(keyword in instruction_lower for keyword in ["write", "create", "generate", "작성", "생성"]):
            return "generation"
        elif any(keyword in instruction_lower for keyword in ["chat", "talk", "conversation", "대화"]):
            return "conversation"
        elif any(keyword in instruction_lower for keyword in ["solve", "calculate", "math", "풀어", "계산"]):
            return "math"
        elif any(keyword in instruction_lower for keyword in ["code", "program", "coding", "코드", "프로그래밍"]):
            return "coding"
        elif any(keyword in instruction_lower for keyword in ["reason", "think", "explain", "추론", "설명"]):
            return "reasoning"
        else:
            return "general"
    
    def _is_high_quality_instruction(self, data: Dict) -> bool:
        """
        명령어 데이터 품질 검사
        
        Args:
            data: 명령어 데이터
            
        Returns:
            고품질 여부
        """
        instruction = data["instruction"]
        output = data["output"]
        
        # 길이 검사
        if (len(instruction) < self.min_instruction_length or 
            len(instruction) > self.max_instruction_length or
            len(output) < self.min_output_length or
            len(output) > self.max_output_length):
            return False
        
        # 의미 없는 패턴 검사
        if self._has_meaningless_patterns(instruction) or self._has_meaningless_patterns(output):
            return False
        
        # 너무 반복적인 응답 검사
        if self._is_too_repetitive(output):
            return False
        
        # 불완전한 응답 검사
        if self._is_incomplete_response(output):
            return False
        
        return True
    
    def _has_meaningless_patterns(self, text: str) -> bool:
        """의미 없는 패턴 검사"""
        # 너무 많은 특수문자
        special_char_ratio = len(re.findall(r'[^\w\s가-힣]', text)) / len(text)
        if special_char_ratio > 0.3:
            return True
        
        # 너무 많은 숫자
        digit_ratio = len(re.findall(r'\d', text)) / len(text)
        if digit_ratio > 0.5:
            return True
        
        # 의미 없는 반복
        words = text.split()
        if len(words) > 3:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                return True
        
        return False
    
    def _is_too_repetitive(self, text: str) -> bool:
        """너무 반복적인 텍스트 검사"""
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) > 2:
            for sentence in sentences:
                if len(sentence) > 10 and text.count(sentence) > 2:
                    return True
        return False
    
    def _is_incomplete_response(self, text: str) -> bool:
        """불완전한 응답 검사"""
        # 너무 짧은 응답
        if len(text.split()) < 3:
            return True
        
        # 끝이 잘린 응답
        if text.endswith(("...", "…", "등등", "등", "그리고")):
            return True
        
        # 의미 없는 응답
        meaningless_responses = [
            "죄송합니다", "모르겠습니다", "확인이 어렵습니다",
            "sorry", "i don't know", "i'm not sure"
        ]
        if any(response in text.lower() for response in meaningless_responses):
            if len(text.split()) < 10:  # 짧으면서 의미 없는 응답
                return True
        
        return False
    
    def _get_instruction_hash(self, data: Dict) -> str:
        """
        명령어 데이터 해시 생성 (중복 제거용)
        
        Args:
            data: 명령어 데이터
            
        Returns:
            MD5 해시
        """
        combined_text = f"{data['instruction']} {data['input']} {data['output']}"
        normalized = re.sub(r'\s+', ' ', combined_text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, data: Dict) -> bool:
        """
        중복 데이터 검사
        
        Args:
            data: 명령어 데이터
            
        Returns:
            중복 여부
        """
        data_hash = self._get_instruction_hash(data)
        if data_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(data_hash)
        return False
    
    def load_instruction_data(self) -> Dict[str, List[Dict]]:
        """
        원시 명령어 데이터 로드
        
        Returns:
            언어별로 그룹화된 명령어 데이터
        """
        logger.info("📂 원시 명령어 데이터 로드 시작")
        
        grouped_data = {"ko": [], "en": []}
        
        # 명령어 데이터 파일들 찾기
        instruction_files = list(self.raw_data_dir.glob("instruction_raw_*.jsonl"))
        
        if not instruction_files:
            logger.warning("❌ 명령어 데이터 파일을 찾을 수 없습니다.")
            return grouped_data
        
        for file_path in instruction_files:
            logger.info(f"처리 중: {file_path.name}")
            
            # 파일명에서 소스 이름 추출
            source_name = file_path.stem.replace("instruction_raw_", "")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(tqdm(f, desc=f"로딩: {file_path.name}")):
                        try:
                            item = json.loads(line.strip())
                            
                            # 명령어 데이터 추출
                            instruction_data = self._extract_instruction_data(item, source_name)
                            if not instruction_data:
                                continue
                            
                            # 품질 검사
                            if not self._is_high_quality_instruction(instruction_data):
                                self.stats["filtered_out"] += 1
                                continue
                            
                            # 중복 검사
                            if self._is_duplicate(instruction_data):
                                self.stats["duplicates_removed"] += 1
                                continue
                            
                            # 언어별 분류
                            language = instruction_data["language"]
                            grouped_data[language].append(instruction_data)
                            self.stats["total_processed"] += 1
                            
                            if language == "ko":
                                self.stats["korean_instructions"] += 1
                            else:
                                self.stats["english_instructions"] += 1
                                
                        except json.JSONDecodeError:
                            logger.warning(f"JSON 파싱 오류: {file_path.name}:{line_num}")
                            continue
                        except Exception as e:
                            logger.warning(f"처리 오류: {file_path.name}:{line_num} - {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"파일 읽기 오류: {file_path.name} - {e}")
                continue
        
        logger.info(f"✅ 원시 명령어 데이터 로드 완료")
        logger.info(f"한국어: {len(grouped_data['ko'])}개, 영어: {len(grouped_data['en'])}개")
        
        return grouped_data
    
    def augment_data(self, grouped_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        데이터 증강
        
        Args:
            grouped_data: 언어별로 그룹화된 데이터
            
        Returns:
            증강된 데이터
        """
        logger.info("🚀 데이터 증강 시작")
        
        augmented_data = {"ko": [], "en": []}
        
        # 기존 데이터 복사
        for lang in ["ko", "en"]:
            augmented_data[lang] = copy.deepcopy(grouped_data[lang])
        
        # 데이터 부족한 언어에 대해 증강 수행
        korean_count = len(grouped_data["ko"])
        english_count = len(grouped_data["en"])
        
        target_korean = int(self.target_count * self.language_ratios["ko"])
        target_english = int(self.target_count * self.language_ratios["en"])
        
        # 한국어 데이터 증강
        if korean_count < target_korean:
            needed = target_korean - korean_count
            logger.info(f"한국어 데이터 증강 필요: {needed}개")
            
            # 기존 데이터에서 변형 생성
            korean_augmented = self._create_variations(grouped_data["ko"], needed, "ko")
            augmented_data["ko"].extend(korean_augmented)
            self.stats["augmented_data"] += len(korean_augmented)
        
        # 영어 데이터 증강
        if english_count < target_english:
            needed = target_english - english_count
            logger.info(f"영어 데이터 증강 필요: {needed}개")
            
            # 기존 데이터에서 변형 생성
            english_augmented = self._create_variations(grouped_data["en"], needed, "en")
            augmented_data["en"].extend(english_augmented)
            self.stats["augmented_data"] += len(english_augmented)
        
        logger.info(f"✅ 데이터 증강 완료")
        logger.info(f"한국어: {len(augmented_data['ko'])}개, 영어: {len(augmented_data['en'])}개")
        
        return augmented_data
    
    def _create_variations(self, data: List[Dict], needed_count: int, language: str) -> List[Dict]:
        """
        데이터 변형 생성
        
        Args:
            data: 원본 데이터
            needed_count: 필요한 데이터 개수
            language: 언어 코드
            
        Returns:
            변형된 데이터 리스트
        """
        variations = []
        
        if not data or needed_count <= 0:
            return variations
        
        # 변형 생성 방법들
        variation_methods = [
            self._paraphrase_instruction,
            self._add_context_variation,
            self._change_formality,
            self._add_examples
        ]
        
        created = 0
        max_attempts = needed_count * 3  # 무한 루프 방지
        attempts = 0
        
        while created < needed_count and attempts < max_attempts:
            # 랜덤하게 원본 데이터 선택
            original = random.choice(data)
            
            # 랜덤하게 변형 방법 선택
            method = random.choice(variation_methods)
            
            try:
                varied = method(original, language)
                if varied and not self._is_duplicate(varied):
                    variations.append(varied)
                    created += 1
                    
            except Exception as e:
                logger.warning(f"변형 생성 오류: {e}")
            
            attempts += 1
        
        logger.info(f"{language} 변형 데이터 {len(variations)}개 생성")
        return variations
    
    def _paraphrase_instruction(self, data: Dict, language: str) -> Optional[Dict]:
        """명령어 패러프레이징"""
        varied = copy.deepcopy(data)
        
        instruction = data["instruction"]
        
        if language == "ko":
            # 한국어 패러프레이징 패턴
            patterns = [
                (r"~하세요", "~해주세요"),
                (r"~하십시오", "~해주세요"),
                (r"설명하세요", "설명해주세요"),
                (r"알려주세요", "설명해주세요"),
                (r"작성하세요", "작성해주세요"),
                (r"무엇", "뭐"),
                (r"어떻게", "어떻게"),
            ]
        else:
            # 영어 패러프레이징 패턴
            patterns = [
                (r"Please ", ""),
                (r"Can you ", ""),
                (r"What is", "What's"),
                (r"How do", "How to"),
                (r"Explain", "Describe"),
                (r"Tell me", "Explain"),
                (r"Write", "Create"),
            ]
        
        # 패턴 적용
        for pattern, replacement in patterns:
            if re.search(pattern, instruction):
                varied["instruction"] = re.sub(pattern, replacement, instruction)
                varied["source"] = data["source"] + "_paraphrased"
                return varied
        
        return None
    
    def _add_context_variation(self, data: Dict, language: str) -> Optional[Dict]:
        """컨텍스트 추가 변형"""
        varied = copy.deepcopy(data)
        
        if language == "ko":
            context_prefixes = [
                "다음 질문에 답해주세요: ",
                "아래 내용에 대해 설명해주세요: ",
                "다음에 대한 정보를 제공해주세요: ",
                "이 문제를 해결해주세요: "
            ]
        else:
            context_prefixes = [
                "Please answer the following question: ",
                "Please explain the following: ",
                "Please provide information about: ",
                "Please solve this problem: "
            ]
        
        prefix = random.choice(context_prefixes)
        varied["instruction"] = prefix + data["instruction"]
        varied["source"] = data["source"] + "_contextualized"
        
        return varied
    
    def _change_formality(self, data: Dict, language: str) -> Optional[Dict]:
        """격식 수준 변경"""
        varied = copy.deepcopy(data)
        
        instruction = data["instruction"]
        
        if language == "ko":
            # 격식체 ↔ 비격식체 변환
            if "요" in instruction or "습니다" in instruction:
                # 격식체 → 비격식체
                patterns = [
                    (r"~습니다", "~다"),
                    (r"~세요", "~어"),
                    (r"해주세요", "해줘"),
                    (r"~입니다", "~이다"),
                ]
            else:
                # 비격식체 → 격식체
                patterns = [
                    (r"~다$", "~습니다"),
                    (r"~어$", "~세요"),
                    (r"해줘", "해주세요"),
                    (r"~이다", "~입니다"),
                ]
        else:
            # 영어는 간단한 변형만
            patterns = [
                (r"can you", "could you"),
                (r"will you", "would you"),
                (r"please", "kindly"),
            ]
        
        # 패턴 적용
        for pattern, replacement in patterns:
            if re.search(pattern, instruction):
                varied["instruction"] = re.sub(pattern, replacement, instruction)
                varied["source"] = data["source"] + "_formality_changed"
                return varied
        
        return None
    
    def _add_examples(self, data: Dict, language: str) -> Optional[Dict]:
        """예시 추가 변형"""
        varied = copy.deepcopy(data)
        
        if language == "ko":
            example_phrases = [
                "예를 들어서 설명해주세요",
                "구체적인 예시와 함께 답해주세요",
                "실제 사례를 포함해서 설명해주세요"
            ]
        else:
            example_phrases = [
                "Please explain with examples",
                "Please provide specific examples",
                "Please include real-world examples"
            ]
        
        example_phrase = random.choice(example_phrases)
        varied["instruction"] = data["instruction"] + ". " + example_phrase
        varied["source"] = data["source"] + "_with_examples"
        
        return varied
    
    def create_mixed_finetuning_dataset(self, grouped_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        한영 혼합 미세조정 데이터셋 생성
        
        Args:
            grouped_data: 언어별로 그룹화된 데이터
            
        Returns:
            혼합된 미세조정 데이터셋
        """
        logger.info("🔀 한영 혼합 미세조정 데이터셋 생성 시작")
        
        korean_data = grouped_data["ko"]
        english_data = grouped_data["en"]
        
        # 언어별 목표 크기 계산
        target_korean = int(self.target_count * self.language_ratios["ko"])
        target_english = int(self.target_count * self.language_ratios["en"])
        
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
        
        # 최소 목표 개수 확인
        if len(mixed_data) < self.min_target_count:
            logger.error(f"❌ 데이터 부족: {len(mixed_data)} < {self.min_target_count}")
            logger.error("데이터 증강 또는 추가 소스 확보가 필요합니다.")
        else:
            logger.info(f"✅ 목표 개수 달성: {len(mixed_data)} >= {self.min_target_count}")
        
        logger.info(f"✅ 혼합 미세조정 데이터셋 생성 완료: {len(mixed_data)}개")
        
        return mixed_data
    
    def save_finetuning_dataset(self, dataset: List[Dict], 
                               dataset_type: str = "mixed") -> str:
        """
        미세조정 데이터셋 저장
        
        Args:
            dataset: 데이터셋
            dataset_type: 데이터셋 타입 (mixed, korean, english)
            
        Returns:
            저장된 파일 경로
        """
        output_file = self.output_dir / f"{dataset_type}_instructions.json"
        
        logger.info(f"💾 미세조정 데이터셋 저장 중: {output_file}")
        
        # 표준 미세조정 형식으로 변환
        finetuning_data = []
        task_distribution = defaultdict(int)
        
        for item in tqdm(dataset, desc="데이터셋 변환"):
            # 태스크 분포 계산
            task_distribution[item["task_category"]] += 1
            
            # 표준 형식으로 변환
            formatted_item = {
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "language": item["language"],
                "task_category": item["task_category"],
                "source": item["source"]
            }
            finetuning_data.append(formatted_item)
        
        # 데이터셋 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(finetuning_data, f, ensure_ascii=False, indent=2)
        
        # 태스크 분포 정보 저장
        distribution_file = self.output_dir / f"{dataset_type}_task_distribution.json"
        with open(distribution_file, 'w', encoding='utf-8') as f:
            json.dump(dict(task_distribution), f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 저장 완료: {output_file}")
        logger.info(f"📊 태스크 분포 저장: {distribution_file}")
        
        return str(output_file)
    
    def save_statistics(self):
        """처리 통계 저장"""
        stats_file = self.output_dir / "finetuning_preprocessing_stats.json"
        
        # 언어별 비율 계산
        if self.stats["total_processed"] > 0:
            korean_ratio = self.stats["korean_instructions"] / self.stats["total_processed"]
            english_ratio = self.stats["english_instructions"] / self.stats["total_processed"]
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
                "min_instruction_length": self.min_instruction_length,
                "max_instruction_length": self.max_instruction_length,
                "min_output_length": self.min_output_length,
                "max_output_length": self.max_output_length
            },
            "targets": {
                "min_target_count": self.min_target_count,
                "target_count": self.target_count,
                "achieved": self.stats["total_processed"] >= self.min_target_count
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 통계 저장 완료: {stats_file}")
    
    def get_processing_summary(self) -> str:
        """처리 요약 정보 반환"""
        korean_ratio = (self.stats["korean_instructions"] / self.stats["total_processed"] 
                       if self.stats["total_processed"] > 0 else 0)
        english_ratio = (self.stats["english_instructions"] / self.stats["total_processed"] 
                        if self.stats["total_processed"] > 0 else 0)
        
        target_achieved = "✅ 달성" if self.stats["total_processed"] >= self.min_target_count else "❌ 미달성"
        
        summary = f"""
📊 미세조정 데이터 전처리 완료 요약
==================================
총 처리된 명령어: {self.stats['total_processed']:,}개
├─ 한국어: {self.stats['korean_instructions']:,}개 ({korean_ratio:.1%})
└─ 영어: {self.stats['english_instructions']:,}개 ({english_ratio:.1%})

목표 달성 여부: {target_achieved}
├─ 최소 목표: {self.min_target_count:,}개
├─ 현재 달성: {self.stats['total_processed']:,}개
└─ 달성률: {(self.stats['total_processed']/self.min_target_count*100):.1f}%

필터링 결과:
├─ 품질 필터로 제외: {self.stats['filtered_out']:,}개
├─ 중복 제거: {self.stats['duplicates_removed']:,}개
└─ 데이터 증강: {self.stats['augmented_data']:,}개

목표 언어 비율:
├─ 한국어: {self.language_ratios['ko']:.1%}
└─ 영어: {self.language_ratios['en']:.1%}

저장 위치: {self.output_dir}
"""
        return summary


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="미세조정 데이터 전처리")
    
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
        "--target-count",
        type=int,
        default=100000,
        help="목표 데이터 개수 (기본값: 100000)"
    )
    parser.add_argument(
        "--min-target",
        type=int,
        default=50000,
        help="최소 목표 데이터 개수 (기본값: 50000)"
    )
    parser.add_argument(
        "--korean-ratio",
        type=float,
        default=0.6,
        help="한국어 비율 (기본값: 0.6)"
    )
    parser.add_argument(
        "--english-ratio",
        type=float,
        default=0.4,
        help="영어 비율 (기본값: 0.4)"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="데이터 증강 비활성화"
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
    
    args = parser.parse_args()
    
    # 언어 비율 검증
    if abs((args.korean_ratio + args.english_ratio) - 1.0) > 0.01:
        logger.error("❌ 한국어와 영어 비율의 합이 1.0이 되어야 합니다.")
        sys.exit(1)
    
    # 전처리기 초기화
    processor = FinetuningDataProcessor(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir
    )
    
    # 설정 업데이트
    processor.target_count = args.target_count
    processor.min_target_count = args.min_target
    processor.language_ratios = {
        "ko": args.korean_ratio,
        "en": args.english_ratio
    }
    
    try:
        # 원시 명령어 데이터 로드
        grouped_data = processor.load_instruction_data()
        
        if not grouped_data["ko"] and not grouped_data["en"]:
            logger.error("❌ 처리할 명령어 데이터가 없습니다.")
            sys.exit(1)
        
        # 데이터 증강 (옵션에 따라)
        if not args.no_augmentation:
            grouped_data = processor.augment_data(grouped_data)
        
        # 데이터셋 생성 및 저장
        if args.korean_only:
            if grouped_data["ko"]:
                logger.info("🇰🇷 한국어 전용 미세조정 데이터셋 생성")
                processor.save_finetuning_dataset(grouped_data["ko"], "korean")
        elif args.english_only:
            if grouped_data["en"]:
                logger.info("🇺🇸 영어 전용 미세조정 데이터셋 생성")
                processor.save_finetuning_dataset(grouped_data["en"], "english")
        else:
            # 혼합 데이터셋 생성
            logger.info("🔀 혼합 미세조정 데이터셋 생성")
            mixed_dataset = processor.create_mixed_finetuning_dataset(grouped_data)
            processor.save_finetuning_dataset(mixed_dataset, "mixed")
        
        # 통계 저장
        processor.save_statistics()
        
        # 요약 정보 출력
        print(processor.get_processing_summary())
        
        logger.info("✅ 미세조정 데이터 전처리 완료!")
        
    except Exception as e:
        logger.error(f"❌ 전처리 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 