#!/bin/bash

# 한국어 소형 언어모델 추론 실행 스크립트 (개선된 버전)
# 사용법: ./run.sh [모델_경로] [디바이스]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 기본 설정
DEFAULT_CHECKPOINT="/data/code/transformers_scratch/outputs/checkpoint-38000"
DEFAULT_DEVICE="auto"

# 함수: 도움말 출력
show_help() {
    echo -e "${BLUE}🇰🇷 한국어 소형 언어모델 추론 시스템 (개선된 토크나이저)${NC}"
    echo "=================================================================="
    echo ""
    echo "사용법:"
    echo "  ./run.sh                              # 기본 체크포인트로 실행 (개선된 토크나이저)"
    echo "  ./run.sh <모델_경로>                  # 특정 모델로 실행"
    echo "  ./run.sh <모델_경로> <디바이스>       # 디바이스 지정하여 실행"
    echo ""
    echo "예시:"
    echo "  ./run.sh                              # checkpoint-38000 사용"
    echo "  ./run.sh ./outputs/checkpoint-12000"
    echo "  ./run.sh /path/to/model cpu"
    echo "  ./run.sh --list                      # 모델 목록만 출력"
    echo "  ./run.sh --help                      # 이 도움말 출력"
    echo ""
    echo "기본값:"
    echo -e "  체크포인트: ${CYAN}${DEFAULT_CHECKPOINT}${NC}"
    echo -e "  디바이스: ${CYAN}${DEFAULT_DEVICE}${NC}"
    echo -e "  토크나이저: ${CYAN}개선된 토크나이저 (132개 한국어 단어 매핑)${NC}"
    echo ""
    echo "디바이스 옵션: auto, cpu, cuda"
    echo "설정 최적화: temperature=1.2, top_p=0.95 (더 나은 텍스트 생성)"
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
    
    # GPU 정보 확인
    if python3 -c "import torch; print('GPU 사용 가능:', torch.cuda.is_available())" | grep -q "True"; then
        echo -e "${GREEN}✅ GPU 가속 사용 가능${NC}"
    else
        echo -e "${YELLOW}⚠️ CPU 모드로 실행됩니다${NC}"
    fi
    
    echo -e "${GREEN}✅ 필수 패키지 확인 완료${NC}"
}

# 함수: 체크포인트 확인
check_checkpoint() {
    local checkpoint_path="$1"
    
    if [[ ! -d "$checkpoint_path" ]]; then
        echo -e "${RED}❌ 체크포인트 디렉토리를 찾을 수 없습니다: $checkpoint_path${NC}"
        return 1
    fi
    
    if [[ ! -f "$checkpoint_path/pytorch_model.bin" ]]; then
        echo -e "${RED}❌ 모델 파일을 찾을 수 없습니다: $checkpoint_path/pytorch_model.bin${NC}"
        return 1
    fi
    
    if [[ ! -f "$checkpoint_path/config.json" ]]; then
        echo -e "${RED}❌ 설정 파일을 찾을 수 없습니다: $checkpoint_path/config.json${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ 체크포인트 확인 완료: $checkpoint_path${NC}"
    return 0
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
        # 인수가 없으면 기본 체크포인트 사용
        echo -e "${BLUE}🚀 기본 체크포인트로 추론 시작${NC}"
        echo -e "${CYAN}모델: $DEFAULT_CHECKPOINT${NC}"
        echo -e "${CYAN}디바이스: $DEFAULT_DEVICE${NC}"
        echo -e "${CYAN}토크나이저: 개선된 토크나이저 (한국어 최적화)${NC}"
        echo ""
        
        if check_checkpoint "$DEFAULT_CHECKPOINT"; then
            python3 console_app.py --checkpoint "$DEFAULT_CHECKPOINT" --device "$DEFAULT_DEVICE"
        else
            echo -e "${YELLOW}기본 체크포인트가 없습니다. 대화형 모드로 전환합니다.${NC}"
            python3 start_inference.py
        fi
    elif [[ "$1" == "--list" || "$1" == "-l" ]]; then
        # 목록 출력 모드
        python3 start_inference.py --list
    elif [[ $# -eq 1 ]]; then
        # 모델 경로만 지정
        echo -e "${BLUE}🚀 모델 실행: $1${NC}"
        if check_checkpoint "$1"; then
            python3 console_app.py --checkpoint "$1" --device "$DEFAULT_DEVICE"
        else
            exit 1
        fi
    elif [[ $# -eq 2 ]]; then
        # 모델 경로와 디바이스 지정
        echo -e "${BLUE}🚀 모델 실행: $1 (디바이스: $2)${NC}"
        if check_checkpoint "$1"; then
            python3 console_app.py --checkpoint "$1" --device "$2"
        else
            exit 1
        fi
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