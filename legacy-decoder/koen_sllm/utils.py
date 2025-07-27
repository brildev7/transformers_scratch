"""
Utility functions for Korean sLLM
한국어 sLLM 유틸리티 함수들
"""
import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging


def set_seed(seed: int = 42):
    """시드 설정으로 재현 가능한 결과 보장"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """사용 가능한 최적의 디바이스 반환"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name()}")
        print(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """모델 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """모델 크기 계산 (MB 단위)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size) / 1024 / 1024  # MB
    
    return {
        'total_mb': total_size,
        'params_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024
    }


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """JSON 파일 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(dirpath: str):
    """디렉토리 생성 (없으면)"""
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """시간을 읽기 쉬운 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}분 {secs:.1f}초"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}시간 {int(minutes)}분 {secs:.1f}초"


def format_number(num: int) -> str:
    """숫자를 읽기 쉬운 형식으로 변환"""
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}B"


class AverageMeter:
    """평균값 계산 유틸리티"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """실행 시간 측정 유틸리티"""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str):
        """타이머 시작"""
        import time
        self.start_times[name] = time.time()
    
    def end(self, name: str) -> float:
        """타이머 종료 및 시간 반환"""
        import time
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.start_times[name]
        
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
        
        del self.start_times[name]
        return elapsed
    
    def get_average(self, name: str) -> float:
        """평균 시간 반환"""
        if name not in self.times:
            return 0.0
        return sum(self.times[name]) / len(self.times[name])
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """타이머 요약 반환"""
        summary = {}
        for name, times in self.times.items():
            summary[name] = {
                'total': sum(times),
                'average': sum(times) / len(times),
                'count': len(times),
                'min': min(times),
                'max': max(times)
            }
        return summary


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """로깅 설정"""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 기본 설정
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    logger = logging.getLogger()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def check_cuda_memory():
    """CUDA 메모리 사용량 체크"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    max_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        'allocated_gb': allocated,
        'cached_gb': cached,
        'max_memory_gb': max_memory,
        'free_gb': max_memory - cached,
        'utilization_percent': (cached / max_memory) * 100
    }


def clear_cuda_cache():
    """CUDA 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA 캐시가 정리되었습니다.")


def calculate_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """모델의 FLOPs 계산 (근사치)"""
    def flop_count(module, input, output):
        if isinstance(module, torch.nn.Linear):
            return module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv2d):
            return (module.in_channels * module.out_channels * 
                   module.kernel_size[0] * module.kernel_size[1] * 
                   output.size(-2) * output.size(-1))
        return 0
    
    model.eval()
    total_flops = 0
    
    def add_flops_hook(module):
        def hook(module, input, output):
            nonlocal total_flops
            total_flops += flop_count(module, input, output)
        return hook
    
    # 훅 등록
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(add_flops_hook(module)))
    
    # 더미 입력으로 순전파
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        model(dummy_input)
    
    # 훅 제거
    for hook in hooks:
        hook.remove()
    
    return total_flops


def print_model_summary(model: torch.nn.Module, input_shape: Optional[tuple] = None):
    """모델 요약 정보 출력"""
    print("=" * 60)
    print("모델 요약")
    print("=" * 60)
    
    # 파라미터 수
    param_count = count_parameters(model)
    print(f"총 파라미터 수: {format_number(param_count['total'])}")
    print(f"학습 가능한 파라미터 수: {format_number(param_count['trainable'])}")
    
    # 모델 크기
    model_size = get_model_size(model)
    print(f"모델 크기: {model_size['total_mb']:.1f} MB")
    
    # FLOPs (입력 형태가 주어진 경우)
    if input_shape:
        try:
            flops = calculate_flops(model, input_shape)
            print(f"FLOPs: {format_number(flops)}")
        except Exception as e:
            print(f"FLOPs 계산 실패: {e}")
    
    # CUDA 메모리 (사용 가능한 경우)
    cuda_info = check_cuda_memory()
    if cuda_info:
        print(f"CUDA 메모리 사용량: {cuda_info['allocated_gb']:.1f} GB")
        print(f"CUDA 메모리 사용률: {cuda_info['utilization_percent']:.1f}%")
    
    print("=" * 60)


def create_model_config_template(save_path: str):
    """모델 설정 템플릿 생성"""
    from .config import ModelConfig
    
    config = ModelConfig()
    config.save(save_path)
    print(f"설정 템플릿이 {save_path}에 생성되었습니다.")


def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """설정 검증"""
    errors = []
    
    required_fields = [
        'vocab_size', 'd_model', 'n_heads', 'n_layers',
        'max_seq_len', 'learning_rate', 'batch_size'
    ]
    
    for field in required_fields:
        if field not in config_dict:
            errors.append(f"필수 필드 누락: {field}")
    
    # 값 범위 검증
    if 'vocab_size' in config_dict and config_dict['vocab_size'] <= 0:
        errors.append("vocab_size는 양수여야 합니다")
    
    if 'd_model' in config_dict and config_dict['n_heads'] in config_dict:
        if config_dict['d_model'] % config_dict['n_heads'] != 0:
            errors.append("d_model은 n_heads로 나누어떨어져야 합니다")
    
    if 'learning_rate' in config_dict and config_dict['learning_rate'] <= 0:
        errors.append("learning_rate는 양수여야 합니다")
    
    return errors


if __name__ == "__main__":
    # 유틸리티 함수 테스트
    print("유틸리티 함수 테스트")
    
    # 디바이스 체크
    device = get_device()
    print(f"선택된 디바이스: {device}")
    
    # CUDA 메모리 체크
    cuda_info = check_cuda_memory()
    if cuda_info:
        print(f"CUDA 메모리 정보: {cuda_info}")
    
    # 타이머 테스트
    timer = Timer()
    timer.start("test")
    import time
    time.sleep(0.1)
    elapsed = timer.end("test")
    print(f"타이머 테스트: {elapsed:.3f}초")
    
    # 숫자 포맷 테스트
    numbers = [123, 1234, 1234567, 1234567890]
    for num in numbers:
        print(f"{num} -> {format_number(num)}")
    
    print("테스트 완료!") 