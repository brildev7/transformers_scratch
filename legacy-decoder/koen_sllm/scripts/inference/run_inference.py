#!/usr/bin/env python3
"""
한국어 소형 언어모델 추론 실행 스크립트

이 스크립트는 학습된 모델을 사용하여 다양한 추론 작업을 수행할 수 있습니다.
"""

import sys
import os
import argparse
from pathlib import Path

# 절대 경로로 모듈 임포트 (현재 디렉토리 기준)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from console_app import main as console_main


def print_banner():
    """프로그램 시작 배너 출력"""
    print("=" * 70)
    print("🇰🇷 한국어 소형 언어모델 (Korean SLLM) 추론 시스템")
    print("=" * 70)
    print()


def check_requirements():
    """필요한 패키지들이 설치되어 있는지 확인"""
    try:
        import torch
        import numpy
        print(f"✅ PyTorch {torch.__version__} 설치됨")
        print(f"✅ CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU 장치: {torch.cuda.get_device_name()}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
    except ImportError as e:
        print(f"❌ 필수 패키지가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치하세요:")
        print("pip install -r requirements.txt")
        sys.exit(1)


def find_available_checkpoints(base_path="./outputs"):
    """사용 가능한 체크포인트 목록 반환"""
    checkpoints = []
    
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            checkpoint_path = os.path.join(base_path, item)
            if (os.path.isdir(checkpoint_path) and 
                item.startswith("checkpoint-") and
                os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) and
                os.path.exists(os.path.join(checkpoint_path, "config.json"))):
                checkpoints.append(checkpoint_path)
    
    return sorted(checkpoints)


def select_checkpoint_interactive():
    """대화형으로 체크포인트 선택"""
    checkpoints = find_available_checkpoints()
    
    if not checkpoints:
        print("❌ 사용 가능한 체크포인트를 찾을 수 없습니다.")
        print("./outputs/ 디렉토리에 체크포인트가 있는지 확인하세요.")
        sys.exit(1)
    
    print("📁 사용 가능한 체크포인트:")
    for i, checkpoint in enumerate(checkpoints, 1):
        # 파일 크기 정보
        model_file = os.path.join(checkpoint, "pytorch_model.bin")
        if os.path.exists(model_file):
            size_gb = os.path.getsize(model_file) / (1024**3)
            print(f"  {i}. {checkpoint} ({size_gb:.1f} GB)")
        else:
            print(f"  {i}. {checkpoint}")
    
    print()
    
    while True:
        try:
            choice = input(f"체크포인트를 선택하세요 (1-{len(checkpoints)}): ").strip()
            
            if not choice:
                # 엔터만 누르면 최신 체크포인트 선택
                return checkpoints[-1]
            
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            else:
                print(f"❌ 1과 {len(checkpoints)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("❌ 올바른 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)


def quick_test(checkpoint_path, device="auto"):
    """빠른 테스트 실행"""
    print("🧪 빠른 테스트 실행 중...")
    
    try:
        from inference import InferenceEngine
        
        # 모델 로드
        engine = InferenceEngine.from_checkpoint(checkpoint_path, device=device)
        
        # 간단한 텍스트 생성 테스트
        test_prompts = ["안녕하세요", "오늘 날씨는", "인공지능의 미래"]
        
        for prompt in test_prompts:
            print(f"\n프롬프트: '{prompt}'")
            response = engine.generate_text(
                prompt=prompt,
                max_length=30,
                temperature=0.8,
                do_sample=True
            )
            print(f"응답: '{response}'")
        
        print("\n✅ 테스트 완료! 모델이 정상적으로 동작합니다.")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="한국어 소형 언어모델 추론 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_inference.py                                    # 대화형 체크포인트 선택
  python run_inference.py --checkpoint ./outputs/checkpoint-12000
  python run_inference.py --test --checkpoint ./outputs/checkpoint-8000
  python run_inference.py --device cpu
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="모델 체크포인트 경로 (지정하지 않으면 대화형 선택)"
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="토크나이저 어휘 파일 경로 (선택적)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="사용할 디바이스"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="빠른 테스트만 실행하고 종료"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="시작 배너 숨기기"
    )
    
    args = parser.parse_args()
    
    # 배너 출력
    if not args.no_banner:
        print_banner()
    
    # 요구사항 체크
    check_requirements()
    
    # 체크포인트 선택
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        
        # 체크포인트 유효성 검사
        if not os.path.exists(checkpoint_path):
            print(f"❌ 체크포인트 경로를 찾을 수 없습니다: {checkpoint_path}")
            
            # 사용 가능한 체크포인트 제안
            available = find_available_checkpoints()
            if available:
                print("\n사용 가능한 체크포인트:")
                for cp in available:
                    print(f"  • {cp}")
            
            sys.exit(1)
        
        # 필수 파일 확인
        model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        config_file = os.path.join(checkpoint_path, "config.json")
        
        if not os.path.exists(model_file):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_file}")
            sys.exit(1)
        
        if not os.path.exists(config_file):
            print(f"❌ 설정 파일을 찾을 수 없습니다: {config_file}")
            sys.exit(1)
    
    else:
        # 대화형 체크포인트 선택
        checkpoint_path = select_checkpoint_interactive()
    
    print(f"📂 선택된 체크포인트: {checkpoint_path}")
    
    # 테스트 모드
    if args.test:
        success = quick_test(checkpoint_path, args.device)
        sys.exit(0 if success else 1)
    
    # 콘솔 애플리케이션 실행
    print("\n🚀 대화형 콘솔을 시작합니다...")
    print("(종료하려면 /exit 또는 Ctrl+C를 누르세요)")
    print()
    
    # sys.argv 수정하여 console_app에 전달
    sys.argv = [
        "console_app.py",
        "--checkpoint", checkpoint_path,
        "--device", args.device
    ]
    
    # 개선된 토크나이저를 기본으로 사용
    if not args.tokenizer:
        sys.argv.extend(["--use-improved-tokenizer"])
    
    sys.argv.extend([
        sys.argv.extend(["--tokenizer", args.tokenizer])
    
    try:
        console_main()
    except KeyboardInterrupt:
        print("\n프로그램이 취소되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 