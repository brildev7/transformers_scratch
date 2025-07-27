#!/bin/bash

# 한국어 sLLM 데이터셋 다운로드 메인 스크립트
# Main dataset download script for Korean sLLM

set -e  # 에러 발생 시 스크립트 종료

# =============================================================================
# 모듈 로드
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/download_core.sh"

# =============================================================================
# 메인 스크립트 함수들
# =============================================================================

print_main_help() {
    echo "한국어 sLLM 데이터셋 다운로드 메인 스크립트"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "🤖 자동 모드 옵션:"
    echo "  --auto           모든 데이터셋을 자동으로 다운로드"
    echo "  -k, --korean     한국어 데이터셋만 다운로드"
    echo "  -e, --english    영어 데이터셋만 다운로드"
    echo "  -p, --pretraining  사전훈련용 데이터셋 다운로드"
    echo "  -t, --finetuning 파인튜닝용 데이터셋 생성"
    echo "  -f, --force      기존 데이터 무시하고 새로 다운로드"
    echo "  -s, --small      소량 샘플만 다운로드 (테스트용)"
    echo ""
    echo "🖱️  대화형 모드:"
    echo "  (옵션 없음)      대화형 메뉴로 단계별 선택"
    echo ""
    echo "🔧 기타 옵션:"
    echo "  -c, --check      디스크 공간만 확인"
    echo "  -h, --help       도움말 표시"
    echo ""
    echo "🔤 개별 모듈 사용:"
    echo "  ./pretraining_datasets.sh   # 사전훈련 데이터셋만"
    echo "  ./finetuning_datasets.sh    # 파인튜닝 데이터셋만"
    echo ""
    echo "예시:"
    echo "  $0               # 대화형 메뉴 시작"
    echo "  $0 --auto        # 모든 데이터셋 자동 다운로드"
    echo "  $0 --pretraining # 사전훈련용 데이터셋만"
    echo "  $0 --force       # 강제 새로 다운로드"
}

# 데이터셋 유형 선택 메뉴
select_dataset_type_menu() {
    echo -e "\n${BLUE}📊 데이터셋 유형 선택${NC}"
    echo ""
    echo "다운로드할 데이터셋 유형을 선택하세요:"
    echo "1) 사전훈련 데이터셋 (한국어/영어 대용량 텍스트)"
    echo "2) 파인튜닝 데이터셋 (명령어-응답 데이터)"
    echo "3) 모두 다운로드"
    echo "4) 취소"
    echo ""
    
    while true; do
        read -p "선택 (1-4): " choice
        case $choice in
            1)
                DATASET_TYPE="pretraining"
                break
                ;;
            2)
                DATASET_TYPE="finetuning"
                break
                ;;
            3)
                DATASET_TYPE="both"
                break
                ;;
            4)
                echo "취소되었습니다."
                exit 0
                ;;
            *)
                echo -e "${RED}❌ 1-4 중에서 선택해주세요.${NC}"
                ;;
        esac
    done
}

# 모듈 실행 함수
execute_module() {
    local module_type="$1"
    local extra_args="$2"
    
    case "$module_type" in
        "pretraining")
            echo -e "${GREEN}🔤 사전훈련 데이터셋 모듈 실행...${NC}"
            bash "$SCRIPT_DIR/pretraining_datasets.sh" $extra_args
            ;;
        "finetuning")
            echo -e "${GREEN}🎯 파인튜닝 데이터셋 모듈 실행...${NC}"
            bash "$SCRIPT_DIR/finetuning_datasets.sh" $extra_args
            ;;
        "both")
            echo -e "${GREEN}🔤 사전훈련 데이터셋 먼저 실행...${NC}"
            bash "$SCRIPT_DIR/pretraining_datasets.sh" $extra_args
            echo ""
            echo -e "${GREEN}🎯 파인튜닝 데이터셋 실행...${NC}"
            bash "$SCRIPT_DIR/finetuning_datasets.sh" $extra_args
            ;;
        *)
            echo -e "${RED}❌ 알 수 없는 모듈 타입: $module_type${NC}"
            exit 1
            ;;
    esac
}

# =============================================================================
# 메인 실행 함수들
# =============================================================================

# 대화형 모드 메인 함수
main_interactive() {
    print_banner
    
    echo -e "${GREEN}🖱️  대화형 모드로 데이터셋 다운로드를 시작합니다!${NC}"
    echo ""
    
    # 환경 확인
    check_environment
    
    # 1단계: 빠른 시작 옵션
    echo -e "${BLUE}⚡ 빠른 시작${NC}"
    echo ""
    echo "모든 데이터셋을 자동으로 다운로드하시겠습니까?"
    echo "1) 예 - 모든 데이터셋 자동 다운로드"
    echo "2) 아니오 - 세부 옵션 선택"
    echo ""
    
    while true; do
        read -p "선택 (1-2): " auto_choice
        case $auto_choice in
            1)
                echo -e "${GREEN}자동 모드로 전환합니다...${NC}"
                execute_module "both" "--auto $FORCE_FLAG $SMALL_FLAG"
                return
                ;;
            2)
                echo -e "${GREEN}대화형 모드를 계속합니다...${NC}"
                break
                ;;
            *)
                echo -e "${RED}❌ 1 또는 2를 선택해주세요.${NC}"
                ;;
        esac
    done
    
    # 2단계: 데이터셋 유형 선택
    select_dataset_type_menu
    
    # 3단계: 모듈 실행
    echo ""
    echo -e "${YELLOW}선택된 모듈을 실행하시겠습니까?${NC}"
    read -p "(y/N): " final_choice
    if [[ ! "$final_choice" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    execute_module "$DATASET_TYPE" "$FORCE_FLAG $SMALL_FLAG"
}

# 자동 모드 메인 함수
main_auto() {
    print_banner
    
    echo -e "${GREEN}🤖 자동 모드로 데이터셋 다운로드를 시작합니다!${NC}"
    echo "다운로드 타입: $DOWNLOAD_TYPE"
    echo ""
    
    if [ "$CHECK_ONLY" = true ]; then
        check_disk_space
        exit 0
    fi
    
    check_environment
    
    # 디스크 공간 확인
    check_disk_space
    
    echo -e "${YELLOW}계속하시겠습니까? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    # 다운로드 타입에 따라 모듈 실행
    case "$DOWNLOAD_TYPE" in
        "korean")
            execute_module "pretraining" "--korean $FORCE_FLAG $SMALL_FLAG"
            ;;
        "english")
            execute_module "pretraining" "--english $FORCE_FLAG $SMALL_FLAG"
            ;;
        "pretraining")
            execute_module "pretraining" "--auto $FORCE_FLAG $SMALL_FLAG"
            ;;
        "finetuning")
            execute_module "finetuning" "--auto $FORCE_FLAG"
            ;;
        "auto")
            execute_module "both" "--auto $FORCE_FLAG $SMALL_FLAG"
            ;;
        *)
            echo -e "${RED}❌ 알 수 없는 다운로드 타입: $DOWNLOAD_TYPE${NC}"
            exit 1
            ;;
    esac
}

# =============================================================================
# 명령행 인수 파싱 및 메인 실행
# =============================================================================

# 변수 초기화
init_core_variables

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            DOWNLOAD_TYPE="auto"
            INTERACTIVE_MODE=false
            shift
            ;;
        -k|--korean)
            DOWNLOAD_TYPE="korean"
            INTERACTIVE_MODE=false
            shift
            ;;
        -e|--english)
            DOWNLOAD_TYPE="english"
            INTERACTIVE_MODE=false
            shift
            ;;
        -p|--pretraining)
            DOWNLOAD_TYPE="pretraining"
            INTERACTIVE_MODE=false
            shift
            ;;
        -t|--finetuning)
            DOWNLOAD_TYPE="finetuning"
            INTERACTIVE_MODE=false
            shift
            ;;
        -f|--force)
            FORCE_FLAG="--force"
            shift
            ;;
        -s|--small)
            SMALL_FLAG="--small"
            shift
            ;;
        -c|--check)
            CHECK_ONLY=true
            INTERACTIVE_MODE=false
            shift
            ;;
        -h|--help)
            print_main_help
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            print_main_help
            exit 1
            ;;
    esac
done

# 메인 실행
if [ "$INTERACTIVE_MODE" = true ]; then
    main_interactive
else
    main_auto
fi 