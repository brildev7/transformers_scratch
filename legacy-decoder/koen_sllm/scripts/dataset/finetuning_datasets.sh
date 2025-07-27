#!/bin/bash

# 한국어 sLLM 지시 미세조정 데이터셋 모듈
# Instruction finetuning datasets module for Korean sLLM

set -e  # 에러 발생 시 스크립트 종료

# =============================================================================
# 핵심 모듈 로드
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/download_core.sh"

# =============================================================================
# 지시 미세조정 데이터셋 관련 함수들
# =============================================================================

# 파인튜닝 데이터셋 선택
select_finetuning_datasets() {
    echo -e "\n${BLUE}🎯 파인튜닝 데이터셋 선택${NC}"
    echo ""
    echo "파인튜닝 데이터셋 옵션:"
    echo "1) 샘플 명령어 데이터 생성 (한국어/영어 각 3개씩)"
    echo "2) 기존 데이터 처리 및 병합"
    echo "3) 둘 다 수행"
    echo ""
    
    while true; do
        read -p "선택 (1-3): " choice
        case $choice in
            1)
                CREATE_SAMPLES=true
                PROCESS_EXISTING=false
                break
                ;;
            2)
                CREATE_SAMPLES=false
                PROCESS_EXISTING=true
                break
                ;;
            3)
                CREATE_SAMPLES=true
                PROCESS_EXISTING=true
                break
                ;;
            *)
                echo -e "${RED}❌ 1-3 중에서 선택해주세요.${NC}"
                ;;
        esac
    done
}

# 파인튜닝 고급 옵션 설정
setup_finetuning_options() {
    echo -e "\n${BLUE}⚙️  파인튜닝 데이터셋 고급 옵션${NC}"
    
    # 샘플 데이터 언어 선택
    if [ "$CREATE_SAMPLES" = true ]; then
        echo ""
        echo "🌐 샘플 데이터 언어 설정:"
        echo "1) 한국어만"
        echo "2) 영어만" 
        echo "3) 한국어 + 영어 (기본)"
        echo ""
        
        while true; do
            read -p "선택 (1-3): " lang_choice
            case $lang_choice in
                1)
                    SAMPLE_LANGUAGES="korean"
                    break
                    ;;
                2)
                    SAMPLE_LANGUAGES="english"
                    break
                    ;;
                3)
                    SAMPLE_LANGUAGES="both"
                    break
                    ;;
                *)
                    echo -e "${RED}❌ 1-3 중에서 선택해주세요.${NC}"
                    ;;
            esac
        done
    fi
    
    # 샘플 데이터 수량 설정
    if [ "$CREATE_SAMPLES" = true ]; then
        echo ""
        echo "📊 샘플 데이터 수량 설정:"
        echo "1) 소량 (각 3개씩) - 테스트용"
        echo "2) 중간 (각 10개씩) - 개발용"
        echo "3) 대량 (각 50개씩) - 실험용"
        echo "4) 직접 입력"
        echo ""
        
        while true; do
            read -p "선택 (1-4): " quantity_choice
            case $quantity_choice in
                1)
                    SAMPLE_COUNT=3
                    break
                    ;;
                2)
                    SAMPLE_COUNT=10
                    break
                    ;;
                3)
                    SAMPLE_COUNT=50
                    break
                    ;;
                4)
                    read -p "생성할 샘플 수를 입력하세요: " custom_count
                    if [[ "$custom_count" =~ ^[0-9]+$ ]] && [ "$custom_count" -gt 0 ]; then
                        SAMPLE_COUNT=$custom_count
                        break
                    else
                        echo -e "${RED}❌ 유효한 숫자를 입력해주세요.${NC}"
                    fi
                    ;;
                *)
                    echo -e "${RED}❌ 1-4 중에서 선택해주세요.${NC}"
                    ;;
            esac
        done
    fi
    
    # 강제 다운로드 여부
    echo ""
    echo "🔄 기존 파일 처리 방식:"
    echo "1) 기존 파일 유지 (덮어쓰지 않음)"
    echo "2) 기존 파일 덮어쓰기"
    echo ""
    
    while true; do
        read -p "선택 (1-2): " force_choice
        case $force_choice in
            1)
                FORCE_FLAG=""
                break
                ;;
            2)
                FORCE_FLAG="--force"
                break
                ;;
            *)
                echo -e "${RED}❌ 1-2 중에서 선택해주세요.${NC}"
                ;;
        esac
    done
}

# 파인튜닝 다운로드 요약 표시
show_finetuning_summary() {
    echo -e "\n${BLUE}📋 파인튜닝 데이터셋 요약${NC}"
    echo "======================================"
    echo -e "저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    echo "🎯 파인튜닝 데이터셋:"
    if [ "$CREATE_SAMPLES" = true ]; then
        echo "  ✅ 샘플 데이터 생성"
        echo "      언어: $SAMPLE_LANGUAGES"
        echo "      수량: 각 ${SAMPLE_COUNT}개"
    else
        echo "  ❌ 샘플 데이터 생성"
    fi
    
    if [ "$PROCESS_EXISTING" = true ]; then
        echo "  ✅ 기존 데이터 처리"
    else
        echo "  ❌ 기존 데이터 처리"
    fi
    echo ""
    
    echo "⚙️  옵션:"
    if [ "$FORCE_FLAG" = "--force" ]; then
        echo "  🔸 기존 파일 덮어쓰기"
    else
        echo "  🔸 기존 파일 유지"
    fi
    echo "======================================"
}

# 파인튜닝 데이터셋 처리 실행
download_finetuning_datasets() {
    echo -e "\n${GREEN}🎯 파인튜닝 데이터셋 처리 중...${NC}"
    
    # 환경 변수 설정 (고유한 파일명 사용 시)
    local extra_args=""
    if [ "$USE_UNIQUE_NAMES" = true ]; then
        extra_args="--unique_names"
    fi
    
    # 샘플 데이터 생성
    if [ "$CREATE_SAMPLES" = true ]; then
        local korean_needed=true
        local english_needed=true
        
        # 언어별 필요 여부 확인
        if [ "$SAMPLE_LANGUAGES" = "korean" ]; then
            english_needed=false
        elif [ "$SAMPLE_LANGUAGES" = "english" ]; then
            korean_needed=false
        fi
        
        # 개별 파일 존재 여부 확인
        if [ "$korean_needed" = true ] && ! should_download_dataset "korean_finetuning" "$OUTPUT_DIR"; then
            korean_needed=false
        fi
        if [ "$english_needed" = true ] && ! should_download_dataset "english_finetuning" "$OUTPUT_DIR"; then
            english_needed=false
        fi
        
        if [ "$korean_needed" = true ] || [ "$english_needed" = true ]; then
            echo -e "${YELLOW}📝 샘플 명령어 데이터 생성...${NC}"
            
            # 언어별 인수 구성
            local lang_args=""
            if [ "$SAMPLE_LANGUAGES" = "korean" ]; then
                lang_args="--korean_only"
            elif [ "$SAMPLE_LANGUAGES" = "english" ]; then
                lang_args="--english_only"
            fi
            
            # 샘플 수량 인수 추가
            local count_args="--sample_count $SAMPLE_COUNT"
            
            python3 "$SCRIPT_DIR/download_finetuning.py" --create_samples $lang_args $count_args $FORCE_FLAG $extra_args
        fi
    fi
    
    # 기존 데이터 처리
    if [ "$PROCESS_EXISTING" = true ]; then
        echo -e "${YELLOW}🔄 기존 데이터 처리 및 병합...${NC}"
        python3 "$SCRIPT_DIR/download_finetuning.py" --process $FORCE_FLAG $extra_args
    fi
}

# 자동 모드 파인튜닝 데이터 처리
download_finetuning_auto() {
    echo -e "${BLUE}자동 모드: 파인튜닝 데이터셋 처리 시작...${NC}"
    
    # 기본값으로 샘플 생성
    echo -e "${GREEN}파인튜닝용 데이터셋 생성 중...${NC}"
    python3 "$SCRIPT_DIR/download_finetuning.py" --create_samples $FORCE_FLAG
}

# 데이터 품질 검증
validate_finetuning_data() {
    echo -e "\n${BLUE}🔍 파인튜닝 데이터 품질 검증${NC}"
    
    local validation_passed=true
    
    # 한국어 데이터 검증
    if [ -f "$OUTPUT_DIR/korean_instructions.json" ]; then
        echo -e "${YELLOW}🇰🇷 한국어 데이터 검증 중...${NC}"
        
        local korean_count=$(python3 -c "
import json, sys
try:
    with open('$OUTPUT_DIR/korean_instructions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 기본 구조 검증
    valid_items = 0
    for item in data:
        if 'instruction' in item and 'output' in item:
            if len(item['instruction'].strip()) > 0 and len(item['output'].strip()) > 0:
                valid_items += 1
    
    print(f'총 {len(data)}개 중 {valid_items}개가 유효함')
    if valid_items == 0:
        sys.exit(1)
except Exception as e:
    print(f'오류: {e}')
    sys.exit(1)
" 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            echo "  ✅ $korean_count"
        else
            echo "  ❌ 한국어 데이터 검증 실패"
            validation_passed=false
        fi
    fi
    
    # 영어 데이터 검증
    if [ -f "$OUTPUT_DIR/english_instructions.json" ]; then
        echo -e "${YELLOW}🇺🇸 영어 데이터 검증 중...${NC}"
        
        local english_count=$(python3 -c "
import json, sys
try:
    with open('$OUTPUT_DIR/english_instructions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 기본 구조 검증
    valid_items = 0
    for item in data:
        if 'instruction' in item and 'output' in item:
            if len(item['instruction'].strip()) > 0 and len(item['output'].strip()) > 0:
                valid_items += 1
    
    print(f'총 {len(data)}개 중 {valid_items}개가 유효함')
    if valid_items == 0:
        sys.exit(1)
except Exception as e:
    print(f'오류: {e}')
    sys.exit(1)
" 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            echo "  ✅ $english_count"
        else
            echo "  ❌ 영어 데이터 검증 실패"
            validation_passed=false
        fi
    fi
    
    if [ "$validation_passed" = true ]; then
        echo -e "${GREEN}✅ 모든 데이터 검증 통과${NC}"
    else
        echo -e "${RED}❌ 일부 데이터 검증 실패${NC}"
        return 1
    fi
}

# =============================================================================
# 메인 실행 함수 (파인튜닝 전용)
# =============================================================================

main_finetuning_interactive() {
    print_banner
    
    echo -e "${GREEN}🎯 지시 미세조정 데이터셋 처리를 시작합니다!${NC}"
    echo ""
    
    # 환경 확인
    check_environment
    
    # 저장 위치 설정
    setup_output_directory
    
    # 파인튜닝 데이터셋 선택
    select_finetuning_datasets
    
    # 고급 옵션 설정
    setup_finetuning_options
    
    # 선택사항 요약
    show_finetuning_summary
    
    # 최종 확인
    echo ""
    echo -e "${YELLOW}위 설정으로 처리를 시작하시겠습니까?${NC}"
    read -p "(y/N): " final_choice
    if [[ ! "$final_choice" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    # 디스크 공간 확인 (파인튜닝은 작은 용량)
    DOWNLOAD_TYPE="finetuning"
    check_disk_space
    
    # 데이터셋 처리 실행
    download_finetuning_datasets
    
    # 데이터 품질 검증
    validate_finetuning_data
    
    # 결과 표시
    show_download_results
}

main_finetuning_auto() {
    print_banner
    
    echo -e "${GREEN}🤖 자동 모드: 파인튜닝 데이터셋 처리${NC}"
    
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
    
    # 파인튜닝은 디스크 용량이 적게 필요
    DOWNLOAD_TYPE="finetuning"
    check_disk_space
    
    download_finetuning_auto
    
    if [ $? -eq 0 ]; then
        validate_finetuning_data
        show_download_results
    else
        echo -e "${RED}❌ 파인튜닝 데이터셋 처리 실패${NC}"
        exit 1
    fi
}

# =============================================================================
# 명령행 인수 파싱 및 메인 실행
# =============================================================================

# 변수 초기화
init_core_variables

# 파인튜닝 전용 변수
SAMPLE_LANGUAGES="both"
SAMPLE_COUNT=3

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            INTERACTIVE_MODE=false
            shift
            ;;
        --samples)
            CREATE_SAMPLES=true
            INTERACTIVE_MODE=false
            shift
            ;;
        --process)
            PROCESS_EXISTING=true
            INTERACTIVE_MODE=false
            shift
            ;;
        --korean-only)
            SAMPLE_LANGUAGES="korean"
            shift
            ;;
        --english-only)
            SAMPLE_LANGUAGES="english"
            shift
            ;;
        --count)
            SAMPLE_COUNT="$2"
            if ! [[ "$SAMPLE_COUNT" =~ ^[0-9]+$ ]] || [ "$SAMPLE_COUNT" -le 0 ]; then
                echo -e "${RED}❌ --count는 양수여야 합니다: $SAMPLE_COUNT${NC}"
                exit 1
            fi
            shift 2
            ;;
        -f|--force)
            FORCE_FLAG="--force"
            shift
            ;;
        -c|--check)
            CHECK_ONLY=true
            INTERACTIVE_MODE=false
            shift
            ;;
        -h|--help)
            echo "지시 미세조정 데이터셋 처리 스크립트"
            echo ""
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --auto           기본 설정으로 자동 처리"
            echo "  --samples        샘플 데이터만 생성"
            echo "  --process        기존 데이터만 처리"
            echo "  --korean-only    한국어 데이터만"
            echo "  --english-only   영어 데이터만"
            echo "  --count N        샘플 수량 지정 (기본: 3)"
            echo "  -f, --force      기존 파일 덮어쓰기"
            echo "  -c, --check      디스크 공간만 확인"
            echo "  -h, --help       도움말 표시"
            echo ""
            echo "예시:"
            echo "  $0                    # 대화형 모드"
            echo "  $0 --auto             # 자동 모드"
            echo "  $0 --samples --count 10  # 샘플 10개씩 생성"
            echo "  $0 --korean-only      # 한국어만"
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            echo "도움말을 보려면 $0 --help를 실행하세요."
            exit 1
            ;;
    esac
done

# 기본값 설정 (명령행에서 지정되지 않은 경우)
if [ -z "$CREATE_SAMPLES" ] && [ -z "$PROCESS_EXISTING" ]; then
    CREATE_SAMPLES=true
    PROCESS_EXISTING=false
fi

# 메인 실행
if [ "$CHECK_ONLY" = true ]; then
    DOWNLOAD_TYPE="finetuning"
    check_disk_space
    exit 0
fi

if [ "$INTERACTIVE_MODE" = true ]; then
    main_finetuning_interactive
else
    main_finetuning_auto
fi 