"""
Inference script for Korean sLLM
한국어 sLLM 추론 스크립트
"""
import torch
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .model import KoreanSLLM
from .tokenizer import KoreanEnglishTokenizer
from .config import ModelConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextGenerator:
    """텍스트 생성 클래스"""
    
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
            self.config = ModelConfig()
        
        # 토크나이저 로드
        self.tokenizer = KoreanEnglishTokenizer(self.config.vocab_size)
        self.tokenizer.load(tokenizer_path)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        logger.info(f"모델이 {self.device}에 로드되었습니다.")
        logger.info(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
    
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
    
    def generate(self,
                prompt: str,
                max_length: int = 100,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9,
                num_return_sequences: int = 1,
                do_sample: bool = True,
                repetition_penalty: float = 1.0) -> List[str]:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_length: 최대 생성 길이
            temperature: 샘플링 온도 (낮을수록 더 결정적)
            top_k: top-k 샘플링
            top_p: nucleus 샘플링 (top-p)
            num_return_sequences: 생성할 시퀀스 수
            do_sample: 샘플링 여부 (False면 greedy)
            repetition_penalty: 반복 패널티
        
        Returns:
            생성된 텍스트 리스트
        """
        # 프롬프트 토큰화
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_texts = []
        
        with torch.no_grad():
            for i in range(num_return_sequences):
                start_time = time.time()
                
                if do_sample:
                    # 샘플링 생성
                    generated = self._generate_with_sampling(
                        input_tensor,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty
                    )
                else:
                    # Greedy 생성
                    generated = self._generate_greedy(
                        input_tensor,
                        max_length=max_length,
                        repetition_penalty=repetition_penalty
                    )
                
                # 디코딩
                generated_text = self.tokenizer.decode(generated[0].tolist())
                
                # 프롬프트 제거
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                generated_texts.append(generated_text)
                
                generation_time = time.time() - start_time
                tokens_per_second = (generated.size(1) - input_tensor.size(1)) / generation_time
                
                logger.info(f"생성 {i+1}/{num_return_sequences} 완료 "
                           f"({tokens_per_second:.1f} tokens/sec)")
        
        return generated_texts
    
    def _generate_with_sampling(self,
                               input_ids: torch.Tensor,
                               max_length: int,
                               temperature: float,
                               top_k: int,
                               top_p: float,
                               repetition_penalty: float) -> torch.Tensor:
        """샘플링을 사용한 생성"""
        current_length = input_ids.size(1)
        generated = input_ids.clone()
        
        for _ in range(max_length - current_length):
            # 현재까지의 시퀀스로 다음 토큰 예측
            with torch.cuda.amp.autocast():
                outputs = self.model(generated)
                logits = outputs['logits']
            
            # 마지막 토큰의 logits만 사용
            next_token_logits = logits[:, -1, :] / temperature
            
            # 반복 패널티 적용
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )
            
            # Top-k 필터링
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Top-p 필터링 (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # top-p를 초과하는 토큰들 제거
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 다음 토큰 샘플링
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 생성된 토큰을 시퀀스에 추가
            generated = torch.cat([generated, next_token], dim=1)
            
            # EOS 토큰이 생성되면 종료
            if next_token.item() == self.tokenizer.special_tokens['<eos>']:
                break
        
        return generated
    
    def _generate_greedy(self,
                        input_ids: torch.Tensor,
                        max_length: int,
                        repetition_penalty: float) -> torch.Tensor:
        """Greedy 디코딩"""
        current_length = input_ids.size(1)
        generated = input_ids.clone()
        
        for _ in range(max_length - current_length):
            # 현재까지의 시퀀스로 다음 토큰 예측
            with torch.cuda.amp.autocast():
                outputs = self.model(generated)
                logits = outputs['logits']
            
            # 마지막 토큰의 logits만 사용
            next_token_logits = logits[:, -1, :]
            
            # 반복 패널티 적용
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )
            
            # 가장 높은 확률의 토큰 선택
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 생성된 토큰을 시퀀스에 추가
            generated = torch.cat([generated, next_token], dim=1)
            
            # EOS 토큰이 생성되면 종료
            if next_token.item() == self.tokenizer.special_tokens['<eos>']:
                break
        
        return generated
    
    def _apply_repetition_penalty(self,
                                 logits: torch.Tensor,
                                 generated: torch.Tensor,
                                 penalty: float) -> torch.Tensor:
        """반복 패널티 적용"""
        for token_id in set(generated[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        
        return logits
    
    def chat(self):
        """대화형 텍스트 생성"""
        print("한국어 sLLM 대화형 생성 모드")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\n프롬프트: ").strip()
                
                if prompt.lower() in ['quit', 'exit', '종료']:
                    print("대화를 종료합니다.")
                    break
                
                if not prompt:
                    continue
                
                print("\n생성 중...")
                generated_texts = self.generate(
                    prompt,
                    max_length=100,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
                print(f"\n생성된 텍스트:\n{generated_texts[0]}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n대화를 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")
                continue
    
    def benchmark(self, prompts: List[str], num_runs: int = 3) -> Dict[str, Any]:
        """성능 벤치마크"""
        logger.info(f"{len(prompts)}개 프롬프트로 벤치마크 시작 (각 {num_runs}회 실행)")
        
        total_times = []
        total_tokens = []
        
        for i, prompt in enumerate(prompts):
            prompt_times = []
            prompt_tokens = []
            
            for run in range(num_runs):
                start_time = time.time()
                
                generated = self.generate(
                    prompt,
                    max_length=50,
                    temperature=1.0,
                    num_return_sequences=1
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 생성된 토큰 수 계산
                generated_tokens = len(self.tokenizer.encode(generated[0]))
                input_tokens = len(self.tokenizer.encode(prompt))
                net_tokens = generated_tokens - input_tokens
                
                prompt_times.append(generation_time)
                prompt_tokens.append(net_tokens)
            
            avg_time = sum(prompt_times) / len(prompt_times)
            avg_tokens = sum(prompt_tokens) / len(prompt_tokens)
            
            total_times.extend(prompt_times)
            total_tokens.extend(prompt_tokens)
            
            logger.info(f"프롬프트 {i+1}: {avg_time:.2f}초, {avg_tokens:.1f} 토큰 "
                       f"({avg_tokens/avg_time:.1f} tokens/sec)")
        
        # 전체 통계
        overall_avg_time = sum(total_times) / len(total_times)
        overall_avg_tokens = sum(total_tokens) / len(total_tokens)
        overall_tokens_per_sec = overall_avg_tokens / overall_avg_time
        
        results = {
            'avg_generation_time': overall_avg_time,
            'avg_tokens_generated': overall_avg_tokens,
            'tokens_per_second': overall_tokens_per_sec,
            'total_runs': len(total_times),
            'device': str(self.device)
        }
        
        logger.info(f"벤치마크 완료: {overall_tokens_per_sec:.1f} tokens/sec")
        
        return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Korean sLLM Inference")
    parser.add_argument("--model_path", type=str, required=True, help="모델 체크포인트 경로")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="토크나이저 경로")
    parser.add_argument("--config_path", type=str, default=None, help="설정 파일 경로")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스")
    parser.add_argument("--mode", type=str, default="single", 
                       choices=["single", "chat", "benchmark"], help="실행 모드")
    
    # 생성 파라미터
    parser.add_argument("--prompt", type=str, default="한국어는", help="입력 프롬프트")
    parser.add_argument("--max_length", type=int, default=100, help="최대 생성 길이")
    parser.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k 샘플링")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 샘플링")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="생성할 시퀀스 수")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="반복 패널티")
    parser.add_argument("--do_sample", action="store_true", help="샘플링 사용")
    
    args = parser.parse_args()
    
    # 텍스트 생성기 초기화
    generator = TextGenerator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
        device=args.device
    )
    
    if args.mode == "single":
        # 단일 생성
        logger.info(f"프롬프트: {args.prompt}")
        
        generated_texts = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty
        )
        
        print(f"\n입력: {args.prompt}")
        print("=" * 50)
        for i, text in enumerate(generated_texts):
            print(f"\n생성 {i+1}:\n{text}")
            print("-" * 50)
    
    elif args.mode == "chat":
        # 대화형 모드
        generator.chat()
    
    elif args.mode == "benchmark":
        # 벤치마크 모드
        benchmark_prompts = [
            "한국어는",
            "인공지능의 발전으로",
            "머신러닝과 딥러닝의",
            "자연어처리 기술은",
            "The future of AI"
        ]
        
        results = generator.benchmark(benchmark_prompts)
        
        print("\n=== 벤치마크 결과 ===")
        print(f"평균 생성 시간: {results['avg_generation_time']:.2f}초")
        print(f"평균 생성 토큰: {results['avg_tokens_generated']:.1f}개")
        print(f"초당 토큰 수: {results['tokens_per_second']:.1f} tokens/sec")
        print(f"총 실행 횟수: {results['total_runs']}회")
        print(f"디바이스: {results['device']}")


if __name__ == "__main__":
    main() 