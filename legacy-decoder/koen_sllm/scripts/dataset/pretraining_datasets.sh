#!/bin/bash

# 한국어 sLLM 사전학습 데이터셋 모듈
# Pretraining datasets module for Korean sLLM

set -e  # 에러 발생 시 스크립트 종료

# =============================================================================
# 핵심 모듈 로드
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/download_core.sh"

# =============================================================================
# 사전학습 데이터셋 관련 함수들
# =============================================================================

# 사전학습 데이터셋 선택
select_pretraining_datasets() {
    local korean_config="$SCRIPT_DIR/../../configs/training/korean_datasets.json"
    local english_config="$SCRIPT_DIR/../../configs/training/english_datasets.json"
    
    echo -e "\n${BLUE}🇰🇷 한국어 사전훈련 데이터셋 선택${NC}"
    
    # 한국어 데이터셋 정보 표시
    if [ -f "$korean_config" ]; then
        echo ""
        echo "사용 가능한 한국어 데이터셋:"
        get_dataset_info "$korean_config" "descriptions"
        echo ""
        echo "모든 한국어 데이터셋을 다운로드하시겠습니까?"
        read -p "(y/N): " korean_choice
        if [[ "$korean_choice" =~ ^[Yy]$ ]]; then
            DOWNLOAD_KOREAN=true
        else
            DOWNLOAD_KOREAN=false
        fi
    else
        echo -e "${RED}❌ 한국어 설정 파일을 찾을 수 없습니다: $korean_config${NC}"
        DOWNLOAD_KOREAN=false
    fi
    
    echo -e "\n${BLUE}🇺🇸 영어 사전훈련 데이터셋 선택${NC}"
    
    # 영어 데이터셋 정보 표시
    if [ -f "$english_config" ]; then
        echo ""
        echo "사용 가능한 영어 데이터셋:"
        get_dataset_info "$english_config" "descriptions"
        echo ""
        echo "모든 영어 데이터셋을 다운로드하시겠습니까?"
        read -p "(y/N): " english_choice
        if [[ "$english_choice" =~ ^[Yy]$ ]]; then
            DOWNLOAD_ENGLISH=true
        else
            DOWNLOAD_ENGLISH=false
        fi
    else
        echo -e "${RED}❌ 영어 설정 파일을 찾을 수 없습니다: $english_config${NC}"
        DOWNLOAD_ENGLISH=false
    fi
    
    # 혼합 데이터셋 생성 여부
    if [ "$DOWNLOAD_KOREAN" = true ] && [ "$DOWNLOAD_ENGLISH" = true ]; then
        echo -e "\n${BLUE}🔀 다국어 혼합 데이터셋${NC}"
        echo "한국어와 영어를 혼합한 데이터셋을 생성하시겠습니까?"
        read -p "(y/N): " mixed_choice
        if [[ "$mixed_choice" =~ ^[Yy]$ ]]; then
            CREATE_MIXED=true
        else
            CREATE_MIXED=false
        fi
    else
        CREATE_MIXED=false
    fi
}

# 사전학습 다운로드 옵션 설정
setup_pretraining_options() {
    echo -e "\n${BLUE}⚙️  사전학습 다운로드 옵션 설정${NC}"
    
    # 소량 다운로드 여부
    echo ""
    echo "테스트용 소량 데이터만 다운로드하시겠습니까?"
    echo "(전체 데이터 대신 각 데이터셋에서 일부만 다운로드)"
    read -p "(y/N): " small_choice
    if [[ "$small_choice" =~ ^[Yy]$ ]]; then
        SMALL_FLAG="--small"
    else
        SMALL_FLAG=""
    fi
    
    # 강제 다운로드 여부
    echo ""
    echo "기존 데이터가 있어도 새로 다운로드하시겠습니까?"
    read -p "(y/N): " force_choice
    if [[ "$force_choice" =~ ^[Yy]$ ]]; then
        FORCE_FLAG="--force"
    else
        FORCE_FLAG=""
    fi
}

# 사전학습 다운로드 요약 표시
show_pretraining_summary() {
    echo -e "\n${BLUE}📋 사전학습 다운로드 요약${NC}"
    echo "======================================"
    echo -e "저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    echo "🔤 사전훈련 데이터셋:"
    if [ "$DOWNLOAD_KOREAN" = true ]; then
        echo "  ✅ 한국어 데이터셋"
    else
        echo "  ❌ 한국어 데이터셋"
    fi
    if [ "$DOWNLOAD_ENGLISH" = true ]; then
        echo "  ✅ 영어 데이터셋"
    else
        echo "  ❌ 영어 데이터셋"
    fi
    if [ "$CREATE_MIXED" = true ]; then
        echo "  ✅ 혼합 데이터셋 생성"
    fi
    echo ""
    
    echo "⚙️  옵션:"
    if [ "$SMALL_FLAG" = "--small" ]; then
        echo "  🔸 소량 테스트 모드"
    else
        echo "  🔸 전체 데이터 다운로드"
    fi
    if [ "$FORCE_FLAG" = "--force" ]; then
        echo "  🔸 기존 데이터 덮어쓰기"
    else
        echo "  🔸 기존 데이터 유지"
    fi
    echo "======================================"
}

# 사전학습 데이터셋 다운로드 실행
download_pretraining_datasets() {
    echo -e "\n${GREEN}📥 사전학습 데이터셋 다운로드 중...${NC}"
    
    # 환경 변수 설정 (고유한 파일명 사용 시)
    local extra_args=""
    if [ "$USE_UNIQUE_NAMES" = true ]; then
        extra_args="--unique_names"
    fi
    
    # 한국어 데이터셋
    if [ "$DOWNLOAD_KOREAN" = true ]; then
        if should_download_dataset "korean_pretraining" "$OUTPUT_DIR"; then
            echo -e "${YELLOW}🇰🇷 한국어 사전훈련 데이터 다운로드...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" --korean_only $FORCE_FLAG $SMALL_FLAG $extra_args
        fi
    fi
    
    # 영어 데이터셋
    if [ "$DOWNLOAD_ENGLISH" = true ]; then
        if should_download_dataset "english_pretraining" "$OUTPUT_DIR"; then
            echo -e "${YELLOW}🇺🇸 영어 사전훈련 데이터 다운로드...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" --english_only $FORCE_FLAG $SMALL_FLAG $extra_args
        fi
    fi
    
    # 혼합 데이터셋
    if [ "$CREATE_MIXED" = true ] && [ "$DOWNLOAD_KOREAN" = true ] && [ "$DOWNLOAD_ENGLISH" = true ]; then
        if should_download_dataset "mixed_pretraining" "$OUTPUT_DIR"; then
            echo -e "${YELLOW}🔀 혼합 데이터셋 생성...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" $FORCE_FLAG $SMALL_FLAG $extra_args
        fi
    fi
}

# 자동 모드 사전학습 다운로드
download_pretraining_auto() {
    echo -e "${BLUE}자동 모드: 사전학습 데이터셋 다운로드 시작...${NC}"
    echo "다운로드 타입: $DOWNLOAD_TYPE"
    
    case "$DOWNLOAD_TYPE" in
        "korean")
            echo -e "${GREEN}한국어 사전훈련 데이터셋 다운로드 중...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" --korean_only $FORCE_FLAG $SMALL_FLAG
            ;;
        "english")
            echo -e "${GREEN}영어 사전훈련 데이터셋 다운로드 중...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" --english_only $FORCE_FLAG $SMALL_FLAG
            ;;
        "pretraining"|"auto")
            echo -e "${GREEN}사전훈련용 데이터셋 다운로드 중...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" $FORCE_FLAG $SMALL_FLAG
            ;;
    esac
}

# =============================================================================
# 메인 실행 함수 (사전학습 전용)
# =============================================================================

main_pretraining_interactive() {
    print_banner
    
    echo -e "${GREEN}🔤 사전학습 데이터셋 다운로드를 시작합니다!${NC}"
    echo ""
    
    # 환경 확인
    check_environment
    
    # 저장 위치 설정
    setup_output_directory
    
    # 사전학습 데이터셋 선택
    select_pretraining_datasets
    
    # 다운로드 옵션 설정
    setup_pretraining_options
    
    # 선택사항 요약
    show_pretraining_summary
    
    # 최종 확인
    echo ""
    echo -e "${YELLOW}위 설정으로 다운로드를 시작하시겠습니까?${NC}"
    read -p "(y/N): " final_choice
    if [[ ! "$final_choice" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    # 디스크 공간 확인
    DOWNLOAD_TYPE="pretraining"
    check_disk_space
    
    # 다운로드 실행
    download_pretraining_datasets
    
    # 결과 표시
    show_download_results
}

main_pretraining_auto() {
    print_banner
    
    echo -e "${GREEN}🤖 자동 모드: 사전학습 데이터셋 다운로드${NC}"
    
    # 기본 설정값 적용
    if [ -z "$OUTPUT_DIR" ]; then
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
        OUTPUT_DIR="$PROJECT_ROOT/datasets"
        mkdir -p "$OUTPUT_DIR"
        mkdir -p "$PROJECT_ROOT/models"
    fi
    
    echo -e "저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    check_environment
    check_disk_space
    
    download_pretraining_auto
    
    if [ $? -eq 0 ]; then
        show_download_results
    else
        echo -e "${RED}❌ 사전학습 데이터셋 다운로드 실패${NC}"
        exit 1
    fi
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
            echo "사전학습 데이터셋 다운로드 스크립트"
            echo ""
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --auto           모든 사전학습 데이터셋을 자동으로 다운로드"
            echo "  -k, --korean     한국어 데이터셋만 다운로드"
            echo "  -e, --english    영어 데이터셋만 다운로드"
            echo "  -f, --force      기존 데이터 무시하고 새로 다운로드"
            echo "  -s, --small      소량 샘플만 다운로드 (테스트용)"
            echo "  -c, --check      디스크 공간만 확인"
            echo "  -h, --help       도움말 표시"
            echo ""
            echo "예시:"
            echo "  $0               # 대화형 모드"
            echo "  $0 --auto        # 모든 사전학습 데이터셋 자동 다운로드"
            echo "  $0 --korean      # 한국어만"
            echo "  $0 --small       # 소량 테스트"
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            echo "도움말을 보려면 $0 --help를 실행하세요."
            exit 1
            ;;
    esac
done

# 메인 실행
if [ "$CHECK_ONLY" = true ]; then
    DOWNLOAD_TYPE="pretraining"
    check_disk_space
    exit 0
fi

if [ "$INTERACTIVE_MODE" = true ]; then
    main_pretraining_interactive
else
    main_pretraining_auto
fi 