#!/usr/bin/env python3
"""
Test setup script for Korean sLLM
한국어 sLLM 설치 및 기본 기능 테스트 스크립트
"""
import sys
import torch
import traceback
from pathlib import Path

def test_imports():
    """모듈 import 테스트"""
    print("=" * 50)
    print("모듈 Import 테스트")
    print("=" * 50)
    
    try:
        from korean_sllm.config import ModelConfig
        print("✅ korean_sllm.config 성공")
        
        from korean_sllm.tokenizer import KoreanEnglishTokenizer
        print("✅ korean_sllm.tokenizer 성공")
        
        from korean_sllm.model import KoreanSLLM
        print("✅ korean_sllm.model 성공")
        
        from korean_sllm.dataset import DatasetDownloader, DatasetManager
        print("✅ korean_sllm.dataset 성공")
        
        from korean_sllm.training import Trainer
        print("✅ korean_sllm.training 성공")
        
        from korean_sllm.validation import ModelValidator
        print("✅ korean_sllm.validation 성공")
        
        from korean_sllm.inference import TextGenerator
        print("✅ korean_sllm.inference 성공")
        
        from korean_sllm.utils import get_device, count_parameters
        print("✅ korean_sllm.utils 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ Import 실패: {e}")
        traceback.print_exc()
        return False


def test_config():
    """설정 시스템 테스트"""
    print("\n" + "=" * 50)
    print("설정 시스템 테스트")
    print("=" * 50)
    
    try:
        from korean_sllm.config import ModelConfig
        
        # 기본 설정 생성
        config = ModelConfig()
        print(f"✅ 기본 설정 생성 성공")
        print(f"   - 어휘 크기: {config.vocab_size}")
        print(f"   - 모델 차원: {config.d_model}")
        print(f"   - 헤드 수: {config.n_heads}")
        print(f"   - 레이어 수: {config.n_layers}")
        
        # 설정 저장/로드 테스트
        test_config_path = "test_config.json"
        config.save(test_config_path)
        loaded_config = ModelConfig.load(test_config_path)
        
        assert config.vocab_size == loaded_config.vocab_size
        assert config.d_model == loaded_config.d_model
        print("✅ 설정 저장/로드 성공")
        
        # 파일 정리
        Path(test_config_path).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ 설정 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_tokenizer():
    """토크나이저 테스트"""
    print("\n" + "=" * 50)
    print("토크나이저 테스트")
    print("=" * 50)
    
    try:
        from korean_sllm.tokenizer import KoreanEnglishTokenizer
        
        # 토크나이저 생성
        tokenizer = KoreanEnglishTokenizer(vocab_size=1000)
        print("✅ 토크나이저 생성 성공")
        
        # 간단한 학습 데이터
        texts = [
            "안녕하세요 Hello world",
            "한국어와 영어를 지원합니다",
            "This is a Korean-English tokenizer",
            "머신러닝과 딥러닝을 공부합니다",
            "Natural language processing is fascinating"
        ]
        
        # 토크나이저 학습
        tokenizer.train(texts)
        print(f"✅ 토크나이저 학습 성공 (어휘 크기: {tokenizer.get_vocab_size()})")
        
        # 인코딩/디코딩 테스트
        test_text = "안녕하세요 Hello"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"   - 원본: {test_text}")
        print(f"   - 인코딩: {encoded[:10]}...")  # 처음 10개만 출력
        print(f"   - 디코딩: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ 토크나이저 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_model():
    """모델 테스트"""
    print("\n" + "=" * 50)
    print("모델 테스트")
    print("=" * 50)
    
    try:
        from korean_sllm.config import ModelConfig
        from korean_sllm.model import KoreanSLLM
        
        # 소형 테스트 설정
        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=1024,
            max_seq_len=128
        )
        
        # 모델 생성
        model = KoreanSLLM(config)
        print(f"✅ 모델 생성 성공")
        print(f"   - 파라미터 수: {model.get_num_params():,}")
        
        # 더미 입력으로 순전파 테스트
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits']
            
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"출력 형태 불일치: {logits.shape} != {expected_shape}"
        
        print(f"✅ 순전파 테스트 성공 (출력 형태: {logits.shape})")
        
        # 생성 테스트
        generated = model.generate(input_ids[:1], max_length=20)
        print(f"✅ 생성 테스트 성공 (생성 길이: {generated.shape[1]})")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_environment():
    """환경 테스트"""
    print("\n" + "=" * 50)
    print("환경 테스트")
    print("=" * 50)
    
    try:
        from korean_sllm.utils import get_device, check_cuda_memory
        
        # 디바이스 체크
        device = get_device()
        print(f"✅ 디바이스: {device}")
        
        # CUDA 메모리 체크 (가능한 경우)
        cuda_info = check_cuda_memory()
        if cuda_info:
            print(f"✅ CUDA 메모리 정보:")
            print(f"   - 총 메모리: {cuda_info['max_memory_gb']:.1f} GB")
            print(f"   - 사용 가능: {cuda_info['free_gb']:.1f} GB")
        else:
            print("⚠️  CUDA 사용 불가 (CPU 모드)")
        
        # PyTorch 버전 체크
        print(f"✅ PyTorch 버전: {torch.__version__}")
        
        # 필수 패키지 체크
        required_packages = [
            'transformers', 'datasets', 'tqdm', 'numpy', 'regex'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} 설치됨")
            except ImportError:
                print(f"❌ {package} 누락")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 환경 테스트 실패: {e}")
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("한국어 sLLM 설치 및 기능 테스트")
    print("=" * 60)
    
    tests = [
        ("Import 테스트", test_imports),
        ("환경 테스트", test_environment),
        ("설정 테스트", test_config),
        ("토크나이저 테스트", test_tokenizer),
        ("모델 테스트", test_model),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} 통과")
            else:
                print(f"\n❌ {test_name} 실패")
        except Exception as e:
            print(f"\n❌ {test_name} 오류: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 한국어 sLLM이 정상적으로 설치되었습니다.")
        print("\n다음 단계:")
        print("1. python common/scripts/train.py --config configs/small_config.json  # 학습 시작")
        print("2. python common/scripts/inference.py --mode chat  # 대화형 추론")
    else:
        print("❌ 일부 테스트가 실패했습니다. 설치를 확인해주세요.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 