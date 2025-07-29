#!/usr/bin/env python3
"""
한국어 소형 언어모델 추론 시작 스크립트

간단한 명령어로 추론을 시작할 수 있는 스크립트입니다.
"""

import sys
import os
import argparse
from pathlib import Path


def resolve_model_path(model_dir):
    """모델 경로 해석 및 해결"""
    # 절대 경로인 경우 그대로 사용
    if os.path.isabs(model_dir):
        return model_dir
    
    # 상대 경로인 경우 여러 위치에서 찾기
    possible_paths = [
        model_dir,  # 현재 디렉토리 기준
        f"../../../../{model_dir}",  # 프로젝트 루트 기준
        f"../../../../../{model_dir}",  # 상위 디렉토리 기준
    ]
    
    # outputs로 시작하는 경우 프로젝트 루트에서도 찾기
    if model_dir.startswith("./outputs/") or model_dir.startswith("outputs/"):
        checkpoint_name = model_dir.split("/")[-1]
        possible_paths.extend([
            f"../../../../outputs/{checkpoint_name}",
            f"../../../../../outputs/{checkpoint_name}",
        ])
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return model_dir  # 찾지 못하면 원래 경로 반환


def validate_model_directory(model_dir):
    """모델 디렉토리 유효성 검사"""
    # 경로 해석
    resolved_path = resolve_model_path(model_dir)
    model_path = Path(resolved_path)
    
    if not model_path.exists():
        print(f"❌ 모델 디렉토리가 존재하지 않습니다: {model_dir}")
        print(f"   (해석된 경로: {resolved_path})")
        return False
    
    # 필요한 파일들 확인
    required_files = ["pytorch_model.bin", "config.json"]
    missing_files = []
    
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ 필수 파일이 없습니다: {', '.join(missing_files)}")
        return False
    
    print(f"✅ 모델 디렉토리 확인: {resolved_path}")
    return True


def find_checkpoints(base_dirs=None):
    """사용 가능한 체크포인트 찾기"""
    if base_dirs is None:
        base_dirs = [
            "./outputs",  # 현재 디렉토리
            "../../../../outputs",  # 프로젝트 루트
            "../../../../../outputs",  # 상위 디렉토리
        ]
    
    checkpoints = []
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    abs_path = str(item.resolve())
                    if abs_path not in checkpoints:
                        # 간단한 유효성 검사 (파일 존재만 확인)
                        if (item / "pytorch_model.bin").exists() and (item / "config.json").exists():
                            checkpoints.append(abs_path)
    
    return sorted(checkpoints)


def list_available_models():
    """사용 가능한 모델 목록 출력"""
    print("🔍 사용 가능한 모델 체크포인트:")
    print("-" * 50)
    
    checkpoints = find_checkpoints()
    if not checkpoints:
        print("사용 가능한 체크포인트가 없습니다.")
        print("./outputs/ 디렉토리에 체크포인트가 있는지 확인하세요.")
        return False
    
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i:2d}. {cp}")
    
    return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="한국어 소형 언어모델 추론 시작 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python start_inference.py                                   # 체크포인트 목록 출력
  python start_inference.py --model ./outputs/checkpoint-12000   # 특정 모델 실행
  python start_inference.py --model /path/to/model --device cpu  # CPU로 실행
  python start_inference.py --list                              # 사용 가능한 모델 목록
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="모델이 저장된 디렉토리 경로"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="사용할 디바이스 (기본값: auto)"
    )
    
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        help="토크나이저 파일 경로 (선택적)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="사용 가능한 모델 목록만 출력"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="빠른 테스트만 실행"
    )
    
    args = parser.parse_args()
    
    # 헤더 출력
    print("=" * 70)
    print("🇰🇷 한국어 소형 언어모델 (Korean SLLM) 추론 시스템")
    print("=" * 70)
    print()
    
    # 목록 출력 모드
    if args.list:
        list_available_models()
        return
    
    # 모델 경로가 지정되지 않은 경우
    if not args.model:
        print("ℹ️  모델 디렉토리가 지정되지 않았습니다.")
        print()
        
        if list_available_models():
            print()
            print("위 목록에서 모델을 선택하려면:")
            print("  python start_inference.py --model <경로>")
        
        return
    
    # 모델 디렉토리 검증
    if not validate_model_directory(args.model):
        print()
        print("💡 도움말:")
        print("  • 모델 디렉토리에는 pytorch_model.bin과 config.json이 있어야 합니다")
        print("  • --list 옵션으로 사용 가능한 모델을 확인할 수 있습니다")
        sys.exit(1)
    
    # 해석된 경로 사용
    resolved_model_path = resolve_model_path(args.model)
    print(f"🖥️  디바이스: {args.device}")
    if args.tokenizer:
        print(f"📝 토크나이저: {args.tokenizer}")
    print()
    
    # 실제 추론 실행
    try:
        # 현재 디렉토리를 sys.path에 추가하여 모듈 import 가능하게 함
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from run_inference import main as run_main
        
        # sys.argv 수정하여 run_inference에 전달 (해석된 경로 사용)
        sys.argv = ["run_inference.py", "--checkpoint", resolved_model_path, "--device", args.device]
        
        if args.tokenizer:
            sys.argv.extend(["--tokenizer", args.tokenizer])
        
        if args.test:
            sys.argv.append("--test")
        
        # 배너 숨기기 (이미 출력했으므로)
        sys.argv.append("--no-banner")
        
        print("🚀 추론 시스템을 시작합니다...")
        print()
        
        run_main()
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 오류: {e}")
        print("필수 패키지가 설치되어 있는지 확인하세요:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 