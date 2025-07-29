"""
한국어 소형 언어모델 추론 엔진
"""

import torch
import time
from typing import Dict, List, Optional, Union
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from model import InferenceModel
from tokenizer import SimpleTokenizer
from training_compatible_model import TrainingCompatibleModel
from training_compatible_tokenizer import TrainingCompatibleTokenizer
from improved_tokenizer import ImprovedTrainingCompatibleTokenizer
from korean_tokenizer import KoreanTokenizer


class InferenceEngine:
    """추론 엔진 클래스"""
    
    def __init__(self, 
                 model,  # InferenceModel 또는 TrainingCompatibleModel
                 tokenizer,  # SimpleTokenizer 또는 TrainingCompatibleTokenizer
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
        
        # 모델 타입 확인
        self.is_training_compatible = isinstance(model, TrainingCompatibleModel)
        if self.is_training_compatible:
            print(f"🔄 학습 호환 모드로 추론 엔진 초기화 (디바이스: {device})")
        else:
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
        
        # 학습 호환 모델인 경우 대화 형식으로 프롬프트 구성
        if self.is_training_compatible:
            # 학습 데이터와 유사한 형식으로 프롬프트 구성
            formatted_prompt = f"사용자: {prompt}\n봇: "
            print(f"프롬프트: '{formatted_prompt}'")
        else:
            formatted_prompt = prompt
            print(f"프롬프트: '{prompt}'")
        
        start_time = time.time()
        
        # 토크나이징
        if self.is_training_compatible:
            # 학습 호환 토크나이저 사용
            encoded = self.tokenizer.encode(
                formatted_prompt, 
                add_special_tokens=True,
                return_tensors="pt"
            )
            input_tensor = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            # 기존 토크나이저 사용
            input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=self.device)
            attention_mask = None
        
        # 배치 크기 확장 (다중 시퀀스 생성용)
        if num_return_sequences > 1:
            input_tensor = input_tensor.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
        
        prompt_length = input_tensor.size(1)
        
        # 텍스트 생성
        self.model.eval()
        with torch.no_grad():
            if self.is_training_compatible:
                # TrainingCompatibleModel의 generate 메서드 사용
                generated_ids = self.model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                # 기존 InferenceModel의 generate 메서드 사용
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
    
    def _generate_raw(self,
                     prompt: str,
                     max_length: int = 100,
                     temperature: float = 1.0,
                     top_k: int = 50,
                     top_p: float = 0.9,
                     do_sample: bool = True,
                     num_return_sequences: int = 1,
                     return_prompt: bool = False) -> Union[str, List[str]]:
        """
        원시 텍스트 생성 (추가 프롬프트 포맷팅 없음)
        """
        
        if not prompt.strip():
            return "" if num_return_sequences == 1 else [""]
        
        print(f"텍스트 생성 중...")
        print(f"프롬프트: '{prompt}'")
        
        start_time = time.time()
        
        # 토크나이징 (포맷팅 없이)
        try:
            if self.is_training_compatible:
                # 학습 호환 토크나이저 사용
                encoded = self.tokenizer.encode(
                    prompt, 
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                if isinstance(encoded, dict):
                    pass  # 디코딩 성공
                input_tensor = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                else:
                    print(f"🔍 attention_mask: None")
            else:
                # 기존 토크나이저 사용
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                input_tensor = torch.tensor([input_ids], device=self.device)
                attention_mask = None
            
            
        except Exception as e:
            print(f"❌ 토크나이징 오류: {e}")
            import traceback
            traceback.print_exc()
            return "" if num_return_sequences == 1 else [""]
        
        # 배치 크기 확장 (다중 시퀀스 생성용)
        if num_return_sequences > 1:
            input_tensor = input_tensor.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
        
        prompt_length = input_tensor.size(1)
        
        # 텍스트 생성
        self.model.eval()
        with torch.no_grad():
            if self.is_training_compatible:
                # TrainingCompatibleModel의 generate 메서드 사용
                generated_ids = self.model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                # 기존 InferenceModel의 generate 메서드 사용
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
        print(f"🔍 chat_generate 호출: is_training_compatible={self.is_training_compatible}")
        
        if self.is_training_compatible:
            # 학습 호환 모드: 대화 컨텍스트 구성
            print(f"✅ 학습 호환 모드로 진입")
            context = ""
            
            if chat_history:
                for turn in chat_history[-5:]:  # 최근 5턴만 사용
                    if turn.get("role") == "user":
                        context += f"사용자: {turn['content']}\n"
                    elif turn.get("role") == "assistant":
                        context += f"봇: {turn['content']}\n"
            
            # 현재 메시지 추가 (generate_text에서 추가 포맷팅하지 않음)
            context += f"사용자: {message}\n봇: "
            
            # 응답 생성 (이미 포맷팅된 context 사용, bypass_formatting=True)
            response = self._generate_raw(
                prompt=context,
                max_length=max_length,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                return_prompt=False
            )
        else:
            # 기존 모드: 단순 메시지 전달
            print(f"❌ 기존 모드로 진입 (학습 호환 아님)")
            response = self.generate_text(
                prompt=message,
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
        # 파라미터 수 계산 방식 결정
        if hasattr(self.model, 'get_num_params'):
            num_params = self.model.get_num_params()
        else:
            num_params = sum(p.numel() for p in self.model.parameters())
        
        # 모델 설정 가져오기
        if self.is_training_compatible and hasattr(self.model, 'config'):
            config = self.model.config
            model_name = config.model_name
            vocab_size = config.vocab_size
            max_pos = config.max_position_embeddings
            hidden_size = config.hidden_size
            num_layers = config.num_layers
            num_heads = config.num_heads
        else:
            model_name = "Korean SLLM"
            vocab_size = getattr(self.tokenizer, 'vocab_size', len(self.tokenizer))
            max_pos = getattr(self.model.config, 'max_position_embeddings', 4096) if hasattr(self.model, 'config') else 4096
            hidden_size = getattr(self.model.config, 'hidden_size', 2048) if hasattr(self.model, 'config') else 2048
            num_layers = getattr(self.model.config, 'num_layers', 24) if hasattr(self.model, 'config') else 24
            num_heads = getattr(self.model.config, 'num_heads', 32) if hasattr(self.model, 'config') else 32
        
        return {
            "model_name": model_name,
            "vocab_size": vocab_size,
            "model_parameters": num_params,
            "device": self.device,
            "max_position_embeddings": max_pos,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads
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
                       device: str = "auto",
                       use_training_compatible: bool = True,
                       use_improved_tokenizer: bool = True,
                       use_korean_tokenizer: bool = False) -> "InferenceEngine":
        """
        체크포인트에서 추론 엔진 로드
        
        Args:
            checkpoint_path: 모델 체크포인트 경로
            tokenizer_path: 토크나이저 어휘 파일 경로 (선택적)
            device: 디바이스
            use_training_compatible: 학습 호환 모델/토크나이저 사용 여부
            use_improved_tokenizer: 개선된 토크나이저 사용 여부
        
        Returns:
            추론 엔진 인스턴스
        """
        print("추론 엔진 로드 중...")
        
        if use_training_compatible:
            print("🔄 학습 호환 모델 및 토크나이저 사용")
            
            # 학습 호환 모델 로드
            try:
                model = TrainingCompatibleModel.from_pretrained(checkpoint_path, device)
                print("✅ 학습 호환 모델 로드 성공")
            except Exception as e:
                print(f"⚠️ 학습 호환 모델 로드 실패: {e}")
                print("기본 InferenceModel로 폴백")
                model = InferenceModel.from_pretrained(checkpoint_path, device)
                use_training_compatible = False
            
            # 토크나이저 선택
            if use_training_compatible:
                if use_korean_tokenizer:
                    print("🇰🇷 한국어 토크나이저 사용")
                    tokenizer = KoreanTokenizer(tokenizer_path)
                    print("✅ 한국어 토크나이저 로드 성공")
                elif use_improved_tokenizer:
                    print("🚀 개선된 토크나이저 사용")
                    tokenizer = ImprovedTrainingCompatibleTokenizer(tokenizer_path)
                    print("✅ 개선된 토크나이저 로드 성공")
                else:
                    print("🔄 기본 학습 호환 토크나이저 사용")
                    tokenizer = TrainingCompatibleTokenizer(tokenizer_path)
                    print("✅ 학습 호환 토크나이저 로드 성공")
            else:
                tokenizer = SimpleTokenizer(tokenizer_path)
        else:
            print("기본 모델 및 토크나이저 사용")
            # 기존 모델 로드
            model = InferenceModel.from_pretrained(checkpoint_path, device)
            
            # 기존 토크나이저 로드
            tokenizer = SimpleTokenizer(tokenizer_path)
        
        return cls(model, tokenizer, device) 