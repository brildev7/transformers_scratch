#!/bin/bash

# 한국어 소형 언어모델 추론 실행 스크립트
# 사용법: ./run.sh [모델_경로] [디바이스]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수: 도움말 출력
show_help() {
    echo -e "${BLUE}🇰🇷 한국어 소형 언어모델 추론 시스템${NC}"
    echo "=================================="
    echo ""
    echo "사용법:"
    echo "  ./run.sh                              # 사용 가능한 모델 목록 출력"
    echo "  ./run.sh <모델_경로>                  # 특정 모델로 추론 실행"
    echo "  ./run.sh <모델_경로> <디바이스>       # 디바이스 지정하여 실행"
    echo ""
    echo "예시:"
    echo "  ./run.sh ./outputs/checkpoint-12000"
    echo "  ./run.sh /path/to/model cpu"
    echo "  ./run.sh --list                      # 모델 목록만 출력"
    echo "  ./run.sh --help                      # 이 도움말 출력"
    echo ""
    echo "디바이스 옵션: auto, cpu, cuda (기본값: auto)"
}

# 함수: 파이썬 환경 확인
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3가 설치되어 있지 않습니다.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Python3 확인됨: $(python3 --version)${NC}"
}

# 함수: 필수 패키지 확인
check_requirements() {
    echo -e "${YELLOW}📦 필수 패키지 확인 중...${NC}"
    
    if ! python3 -c "import torch" &> /dev/null; then
        echo -e "${RED}❌ PyTorch가 설치되어 있지 않습니다.${NC}"
        echo "다음 명령어로 설치하세요:"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 필수 패키지 확인 완료${NC}"
}

# 메인 실행 부분
main() {
    # 도움말 처리
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # 파이썬 환경 확인
    check_python
    check_requirements
    
    echo ""
    
    # 인수에 따른 실행
    if [[ $# -eq 0 ]]; then
        # 인수가 없으면 모델 목록 출력
        echo -e "${BLUE}📋 사용 가능한 모델 확인 중...${NC}"
        python3 start_inference.py
    elif [[ "$1" == "--list" || "$1" == "-l" ]]; then
        # 목록 출력 모드
        python3 start_inference.py --list
    elif [[ $# -eq 1 ]]; then
        # 모델 경로만 지정
        echo -e "${BLUE}🚀 모델 실행: $1${NC}"
        python3 start_inference.py --model "$1"
    elif [[ $# -eq 2 ]]; then
        # 모델 경로와 디바이스 지정
        echo -e "${BLUE}🚀 모델 실행: $1 (디바이스: $2)${NC}"
        python3 start_inference.py --model "$1" --device "$2"
    else
        echo -e "${RED}❌ 잘못된 인수입니다.${NC}"
        echo ""
        show_help
        exit 1
    fi
}

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# 메인 함수 실행
main "$@" 