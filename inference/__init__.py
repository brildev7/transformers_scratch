"""
Korean SLLM 추론 모듈

이 패키지는 학습된 한국어 소형 언어모델의 추론을 위한 모듈들을 포함합니다.
"""

from .model import InferenceModel
from .inference_engine import InferenceEngine
from .console_app import ConsoleApp

__version__ = "1.0.0"
__all__ = ["InferenceModel", "InferenceEngine", "ConsoleApp"] 