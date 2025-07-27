"""
Model validation script for Korean sLLM
한국어 sLLM 모델 검증 스크립트
"""
import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from .model import KoreanSLLM
from .tokenizer import KoreanEnglishTokenizer
from .config import ModelConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """모델 검증 클래스"""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: str,
                 config_path: Optional[str] = None,
                 device: str = "cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 설정 로드
        if config_path:
            self.config = ModelConfig.load(config_path)
        else:
            # 기본 설정 사용
            self.config = ModelConfig()
        
        # 토크나이저 로드
        self.tokenizer = KoreanEnglishTokenizer(self.config.vocab_size)
        self.tokenizer.load(tokenizer_path)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        logger.info(f"모델이 {self.device}에 로드되었습니다.")
    
    def _load_model(self, model_path: str) -> KoreanSLLM:
        """모델 로드"""
        model = KoreanSLLM(self.config).to(self.device)
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # DDP 가중치인 경우 'module.' 제거
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> float:
        """텍스트 리스트에 대한 perplexity 계산"""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
                batch_texts = texts[i:i + batch_size]
                batch_losses = []
                batch_token_counts = []
                
                for text in batch_texts:
                    # 텍스트 토큰화
                    token_ids = self.tokenizer.encode(text)
                    
                    # 너무 짧은 텍스트 스킵
                    if len(token_ids) < 10:
                        continue
                    
                    # 최대 길이로 자르기
                    if len(token_ids) > self.config.max_seq_len:
                        token_ids = token_ids[:self.config.max_seq_len]
                    
                    input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
                    
                    # 모델 예측
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    loss = outputs['loss']
                    
                    # 실제 토큰 수 (패딩 제외)
                    actual_tokens = (input_ids != self.tokenizer.special_tokens['<pad>']).sum().item()
                    
                    batch_losses.append(loss.item() * actual_tokens)
                    batch_token_counts.append(actual_tokens)
                
                if batch_losses:
                    total_loss += sum(batch_losses)
                    total_tokens += sum(batch_token_counts)
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def generate_text(self, 
                      prompt: str,
                      max_length: int = 100,
                      temperature: float = 1.0,
                      top_k: int = 50,
                      top_p: float = 0.9,
                      num_return_sequences: int = 1) -> List[str]:
        """텍스트 생성"""
        # 프롬프트 토큰화
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_texts = []
        
        with torch.no_grad():
            for _ in range(num_return_sequences):
                # 텍스트 생성
                generated = self.model.generate(
                    input_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.special_tokens['<pad>'],
                    eos_token_id=self.tokenizer.special_tokens['<eos>']
                )
                
                # 디코딩
                generated_text = self.tokenizer.decode(generated[0].tolist())
                generated_texts.append(generated_text)
        
        return generated_texts
    
    def evaluate_generation_quality(self, prompts: List[str]) -> Dict[str, float]:
        """생성 품질 평가"""
        results = {
            'avg_length': 0.0,
            'repetition_ratio': 0.0,
            'diversity_score': 0.0
        }
        
        all_generated = []
        total_length = 0
        repetition_scores = []
        
        for prompt in tqdm(prompts, desc="Evaluating generation quality"):
            generated_texts = self.generate_text(prompt, max_length=50, num_return_sequences=3)
            
            for text in generated_texts:
                all_generated.append(text)
                total_length += len(text.split())
                
                # 반복 비율 계산
                words = text.split()
                if len(words) > 1:
                    unique_words = len(set(words))
                    repetition_ratio = 1.0 - (unique_words / len(words))
                    repetition_scores.append(repetition_ratio)
        
        # 평균 길이
        results['avg_length'] = total_length / len(all_generated) if all_generated else 0
        
        # 평균 반복 비율
        results['repetition_ratio'] = np.mean(repetition_scores) if repetition_scores else 0
        
        # 다양성 점수 (유니크한 n-gram 비율)
        all_text = ' '.join(all_generated)
        words = all_text.split()
        if len(words) > 2:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            unique_bigrams = len(set(bigrams))
            diversity_score = unique_bigrams / len(bigrams) if bigrams else 0
            results['diversity_score'] = diversity_score
        
        return results
    
    def run_comprehensive_evaluation(self, 
                                   test_texts: Optional[List[str]] = None,
                                   test_prompts: Optional[List[str]] = None) -> Dict[str, any]:
        """종합적인 모델 평가"""
        logger.info("종합 평가 시작...")
        
        results = {}
        
        # 기본 테스트 데이터
        if test_texts is None:
            test_texts = [
                "한국어는 한반도와 그 부속 도서, 중국 동북부와 러시아 연해주 그리고 일본, 미국 등 세계 각지의 한민족 거주 지역에서 쓰이는 언어이다.",
                "머신러닝은 인공지능의 한 분야로, 컴퓨터가 학습할 수 있도록 하는 알고리즘과 기법을 다룬다.",
                "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
                "Deep learning has revolutionized the field of artificial intelligence in recent years.",
                "자연어처리 기술은 컴퓨터가 인간의 언어를 이해하고 처리할 수 있게 하는 기술이다."
            ]
        
        if test_prompts is None:
            test_prompts = [
                "한국어는",
                "인공지능의 발전으로",
                "Natural language processing",
                "The future of AI",
                "머신러닝과 딥러닝의 차이점은"
            ]
        
        # 1. Perplexity 계산
        logger.info("Perplexity 계산 중...")
        perplexity = self.calculate_perplexity(test_texts)
        results['perplexity'] = perplexity
        logger.info(f"Perplexity: {perplexity:.2f}")
        
        # 2. 생성 품질 평가
        logger.info("생성 품질 평가 중...")
        generation_quality = self.evaluate_generation_quality(test_prompts)
        results['generation_quality'] = generation_quality
        logger.info(f"생성 품질: {generation_quality}")
        
        # 3. 샘플 생성
        logger.info("샘플 텍스트 생성 중...")
        sample_generations = {}
        for prompt in test_prompts[:3]:  # 처음 3개 프롬프트만
            generated = self.generate_text(prompt, max_length=50)
            sample_generations[prompt] = generated[0]
        
        results['sample_generations'] = sample_generations
        
        # 4. 모델 통계
        model_stats = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'vocab_size': self.tokenizer.get_vocab_size(),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }
        results['model_stats'] = model_stats
        
        logger.info("종합 평가 완료!")
        return results
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """평가 결과 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"평가 결과가 {output_path}에 저장되었습니다.")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Korean sLLM Model Validation")
    parser.add_argument("--model_path", type=str, required=True, help="모델 체크포인트 경로")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="토크나이저 경로")
    parser.add_argument("--config_path", type=str, default=None, help="설정 파일 경로")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", help="결과 저장 경로")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스")
    
    args = parser.parse_args()
    
    # 검증기 초기화
    validator = ModelValidator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
        device=args.device
    )
    
    # 종합 평가 실행
    results = validator.run_comprehensive_evaluation()
    
    # 결과 저장
    validator.save_evaluation_results(results, args.output_path)
    
    # 결과 출력
    print("\n=== 평가 결과 ===")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"평균 생성 길이: {results['generation_quality']['avg_length']:.2f} 단어")
    print(f"반복 비율: {results['generation_quality']['repetition_ratio']:.4f}")
    print(f"다양성 점수: {results['generation_quality']['diversity_score']:.4f}")
    print(f"모델 크기: {results['model_stats']['model_size_mb']:.2f} MB")
    print(f"파라미터 수: {results['model_stats']['total_parameters']:,}")
    
    print("\n=== 샘플 생성 ===")
    for prompt, generated in results['sample_generations'].items():
        print(f"프롬프트: {prompt}")
        print(f"생성: {generated}")
        print("-" * 50)


if __name__ == "__main__":
    main() 