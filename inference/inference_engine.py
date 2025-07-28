"""
한국어 소형 언어모델 추론 엔진
"""

import torch
import time
from typing import Dict, List, Optional, Union
from .model import InferenceModel
from .tokenizer import SimpleTokenizer


class InferenceEngine:
    """추론 엔진 클래스"""
    
    def __init__(self, 
                 model: InferenceModel,
                 tokenizer: SimpleTokenizer,
                 device: str = "auto"):
        """
        Args:
            model: 추론용 모델
            tokenizer: 토크나이저
            device: 디바이스 ("auto", "cpu", "cuda")
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        print(f"추론 엔진 초기화 완료 (디바이스: {device})")
    
    def generate_text(self,
                     prompt: str,
                     max_length: int = 100,
                     temperature: float = 1.0,
                     top_k: int = 50,
                     top_p: float = 0.9,
                     do_sample: bool = True,
                     num_return_sequences: int = 1,
                     return_prompt: bool = False) -> Union[str, List[str]]:
        """
        프롬프트에서 텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_length: 최대 생성 길이
            temperature: 샘플링 온도
            top_k: top-k 샘플링
            top_p: nucleus 샘플링
            do_sample: 샘플링 여부
            num_return_sequences: 반환할 시퀀스 수
            return_prompt: 프롬프트를 포함하여 반환할지 여부
        
        Returns:
            생성된 텍스트 (단일 시퀀스면 str, 다중이면 List[str])
        """
        if not prompt.strip():
            return "" if num_return_sequences == 1 else [""]
        
        print(f"텍스트 생성 중...")
        print(f"프롬프트: '{prompt}'")
        
        start_time = time.time()
        
        # 프롬프트 토크나이징
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # 배치 크기 확장 (다중 시퀀스 생성용)
        if num_return_sequences > 1:
            input_tensor = input_tensor.repeat(num_return_sequences, 1)
        
        prompt_length = input_tensor.size(1)
        
        # 텍스트 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 프롬프트 제거 (선택적)
        if not return_prompt:
            generated_ids = generated_ids[:, prompt_length:]
        
        # 디코딩
        if num_return_sequences == 1:
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_per_second = len(generated_ids[0]) / generation_time if generation_time > 0 else 0
            
            print(f"생성 완료 (시간: {generation_time:.2f}초, 속도: {tokens_per_second:.1f} 토큰/초)")
            print(f"생성된 텍스트: '{generated_text}'")
            
            return generated_text
        else:
            generated_texts = []
            for i in range(num_return_sequences):
                text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                generated_texts.append(text)
            
            end_time = time.time()
            generation_time = end_time - start_time
            avg_tokens = sum(len(seq) for seq in generated_ids) / num_return_sequences
            tokens_per_second = avg_tokens / generation_time if generation_time > 0 else 0
            
            print(f"생성 완료 (시간: {generation_time:.2f}초, 평균 속도: {tokens_per_second:.1f} 토큰/초)")
            for i, text in enumerate(generated_texts):
                print(f"생성된 텍스트 {i+1}: '{text}'")
            
            return generated_texts
    
    def complete_text(self, 
                     incomplete_text: str,
                     max_completion_length: int = 50,
                     temperature: float = 0.8,
                     top_k: int = 50,
                     top_p: float = 0.9) -> str:
        """
        불완전한 텍스트 완성
        
        Args:
            incomplete_text: 완성할 텍스트
            max_completion_length: 최대 완성 길이
            temperature: 샘플링 온도
            top_k: top-k 샘플링
            top_p: nucleus 샘플링
        
        Returns:
            완성된 텍스트
        """
        print(f"텍스트 완성 중: '{incomplete_text}'")
        
        completed = self.generate_text(
            prompt=incomplete_text,
            max_length=max_completion_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            return_prompt=False
        )
        
        full_text = incomplete_text + completed
        print(f"완성된 텍스트: '{full_text}'")
        
        return full_text
    
    def chat_generate(self,
                     message: str,
                     chat_history: Optional[List[Dict[str, str]]] = None,
                     max_length: int = 150,
                     temperature: float = 0.9) -> str:
        """
        대화형 텍스트 생성
        
        Args:
            message: 사용자 메시지
            chat_history: 이전 대화 히스토리
            max_length: 최대 생성 길이
            temperature: 샘플링 온도
        
        Returns:
            생성된 응답
        """
        # 대화 컨텍스트 구성
        context = ""
        
        if chat_history:
            for turn in chat_history[-5:]:  # 최근 5턴만 사용
                if turn.get("role") == "user":
                    context += f"사용자: {turn['content']}\n"
                elif turn.get("role") == "assistant":
                    context += f"봇: {turn['content']}\n"
        
        # 현재 메시지 추가
        context += f"사용자: {message}\n봇: "
        
        # 응답 생성
        response = self.generate_text(
            prompt=context,
            max_length=max_length,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            return_prompt=False
        )
        
        # 응답에서 불필요한 부분 제거
        response = response.split('\n')[0].strip()  # 첫 번째 줄만 사용
        
        return response
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """모델 정보 반환"""
        return {
            "model_name": "Korean SLLM",
            "vocab_size": self.tokenizer.vocab_size,
            "model_parameters": self.model.get_num_params(),
            "device": self.device,
            "max_position_embeddings": self.model.config.max_position_embeddings,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_layers,
            "num_heads": self.model.config.num_heads
        }
    
    def benchmark(self, prompt: str = "안녕하세요", num_runs: int = 5) -> Dict[str, float]:
        """성능 벤치마크"""
        print(f"성능 벤치마크 실행 중 ({num_runs}회)...")
        
        times = []
        token_counts = []
        
        for i in range(num_runs):
            print(f"  실행 {i+1}/{num_runs}")
            
            start_time = time.time()
            result = self.generate_text(
                prompt=prompt,
                max_length=50,
                temperature=1.0,
                do_sample=True
            )
            end_time = time.time()
            
            generation_time = end_time - start_time
            token_count = len(self.tokenizer.encode(result))
            
            times.append(generation_time)
            token_counts.append(token_count)
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        avg_tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        benchmark_results = {
            "average_time_seconds": avg_time,
            "average_tokens_generated": avg_tokens,
            "average_tokens_per_second": avg_tokens_per_second,
            "min_time_seconds": min(times),
            "max_time_seconds": max(times)
        }
        
        print("벤치마크 결과:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value:.3f}")
        
        return benchmark_results
    
    @classmethod
    def from_checkpoint(cls,
                       checkpoint_path: str,
                       tokenizer_path: Optional[str] = None,
                       device: str = "auto") -> "InferenceEngine":
        """
        체크포인트에서 추론 엔진 로드
        
        Args:
            checkpoint_path: 모델 체크포인트 경로
            tokenizer_path: 토크나이저 어휘 파일 경로 (선택적)
            device: 디바이스
        
        Returns:
            추론 엔진 인스턴스
        """
        print("추론 엔진 로드 중...")
        
        # 모델 로드
        model = InferenceModel.from_pretrained(checkpoint_path, device)
        
        # 토크나이저 로드
        tokenizer = SimpleTokenizer(tokenizer_path)
        
        return cls(model, tokenizer, device) 