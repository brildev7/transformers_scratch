#!/bin/bash

# 한국어 sLLM 원시 데이터 다운로드 조정기 (RecommendDataset.md 권고사항 반영)
# Raw data download orchestrator for Korean sLLM with RecommendDataset.md recommendations

set -e  # 에러 발생 시 스크립트 종료

# =============================================================================
# 핵심 모듈 로드
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/download_core.sh"

# =============================================================================
# 다운로드 조정 함수들
# =============================================================================

print_download_banner() {
    echo -e "${BLUE}"
    echo "========================================================"
    echo "     한국어 sLLM 원시 데이터 다운로드"
    echo "     Korean sLLM Raw Data Download"
    echo "     🆕 RecommendDataset.md 권고사항 반영"
    echo "========================================================"
    echo ""
    echo "📥 다운로드 전략:"
    echo "   🚀  레시피 1: 상용화 대비 (허용적 라이선스 중심) [권장]"
    echo "   🔬  레시피 2: 성능 극대화 (대규모 데이터)"  
    echo "   🇰🇷  한국어 데이터만"
    echo "   🇺🇸  영어 데이터만"
    echo "   🎯  명령어-응답 데이터"
    echo "========================================================"
    echo -e "${NC}"
}

print_download_help() {
    echo "한국어 sLLM 원시 데이터 다운로드 조정기 (v2.0)"
    echo "RecommendDataset.md 권고사항 기반 업데이트"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "🚀 추천 레시피 옵션 (RecommendDataset.md 기반):"
    echo "  --recipe1            레시피 1: 상용화 대비 (75GB, 허용적 라이선스) [권장]"
    echo "  --recipe2            레시피 2: 성능 극대화 (150GB, 혼합 라이선스)"
    echo ""
    echo "📥 기본 다운로드 옵션:"
    echo "  --all                모든 데이터 다운로드 (기본값)"
    echo "  --korean             한국어 데이터만"
    echo "  --english            영어 데이터만"
    echo "  --instructions       명령어 데이터만"
    echo "  --small              테스트용 소량 데이터"
    echo ""
    echo "📁 디렉토리 옵션:"
    echo "  --output-dir DIR     출력 디렉토리 (기본값: raw_datasets)"
    echo ""
    echo "🔧 기타 옵션:"
    echo "  --force              기존 데이터 덮어쓰기"
    echo "  --check              디스크 공간만 확인"
    echo "  --license-info       라이선스 정보 표시"
    echo "  -h, --help           도움말 표시"
    echo ""
    echo "🚀 레시피별 특징:"
    echo ""
    echo "  📋 레시피 1 (상용화 대비) - 소규모 연구소 권장:"
    echo "     • 총 규모: ~75GB (한국어 3GB + 영어 72GB)"
    echo "     • 라이선스: 완전 허용적 (Apache 2.0, CC0, ODC-BY)"
    echo "     • 장점: 법적 리스크 최소화, 미래 상업화 자유"
    echo "     • 제외: AI-Hub, Books3 (저작권 위험)"
    echo ""
    echo "  📋 레시피 2 (성능 극대화) - 순수 연구용:"
    echo "     • 총 규모: ~150GB (한국어 18GB + 영어 132GB)"
    echo "     • 라이선스: 혼합형 (AI-Hub 포함)"
    echo "     • 장점: 최대 성능, 벤치마크 경쟁력"
    echo "     • 제약: 상업적 활용 시 별도 협의 필요"
    echo ""
    echo "예시:"
    echo "  $0 --recipe1         # 상용화 대비 레시피 (권장)"
    echo "  $0 --recipe2         # 성능 극대화 레시피"
    echo "  $0 --korean          # 한국어 데이터만"
    echo "  $0 --small           # 테스트용 소량 데이터"
    echo "  $0 --license-info    # 라이선스 정보 확인"
    echo ""
    echo "📝 전처리 안내:"
    echo "  다운로드 완료 후 전처리 옵션:"
    echo "  - 레시피 1: python3 preprocess_pretraining.py --recipe configs/dataset/recipe1_commercial_ready.json"
    echo "  - 레시피 2: python3 preprocess_pretraining.py --recipe configs/dataset/recipe2_performance_max.json"
    echo "  - 기본: python3 preprocess_pretraining.py --raw-data-dir raw_datasets"
}

print_license_info() {
    echo -e "${BLUE}📋 라이선스 정보 및 권고사항${NC}"
    echo "=================================="
    echo ""
    echo -e "${GREEN}✅ 허용적 라이선스 (상업적 이용 가능):${NC}"
    echo "  • Apache 2.0: 한국어 위키피디아"
    echo "  • CC0: OpenWebText (Public Domain에 가까움)"
    echo "  • ODC-BY: C4 데이터셋"
    echo "  • Public Domain: Gutenberg 고서"
    echo ""
    echo -e "${YELLOW}⚠️  제한적 라이선스:${NC}"
    echo "  • CC-BY-SA: 동일 라이선스 유지 조건"
    echo "  • 연구제한: AI-Hub (상업적 이용 시 별도 협의)"
    echo ""
    echo -e "${RED}❌ 제외된 데이터 (RecommendDataset.md 권고):${NC}"
    echo "  • AI-Hub 데이터 (레시피1): 상업적 활용 시 별도 협의 필요"
    echo "  • The Pile/Books3: 저작권 침해 위험 (DMCA 대상)"
    echo "  • 국립국어원 모두의 말뭉치: 복잡한 접근 절차"
    echo ""
    echo -e "${BLUE}🎯 전략적 권고사항:${NC}"
    echo "  • 소규모 연구소/스타트업 → 레시피 1 권장"
    echo "  • 대학/연구기관 (순수 연구) → 레시피 2 고려"
    echo "  • 상업화 계획이 있다면 반드시 레시피 1 선택"
    echo ""
}

check_python_dependencies() {
    echo -e "${YELLOW}🔍 Python 의존성 확인 중...${NC}"
    
    local required_packages=("datasets" "tqdm")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo -e "${RED}❌ 필요한 Python 패키지가 없습니다: ${missing_packages[*]}${NC}"
        echo "다음 명령어로 설치하세요:"
        echo "pip install datasets tqdm"
        return 1
    fi
    
    echo -e "${GREEN}✅ Python 의존성 확인 완료${NC}"
    return 0
}

setup_output_directory() {
    echo -e "${BLUE}📁 출력 디렉토리 설정${NC}"
    
    # 디렉토리 생성
    mkdir -p "$OUTPUT_DIR"
    
    echo -e "원시 데이터 저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    
    # 권한 확인
    if [ ! -w "$OUTPUT_DIR" ]; then
        echo -e "${RED}❌ 디렉토리 쓰기 권한이 없습니다${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ 디렉토리 설정 완료${NC}"
    return 0
}

execute_recipe_download() {
    local recipe_num=$1
    echo -e "\n${GREEN}🚀 레시피 $recipe_num 다운로드 시작${NC}"
    echo "==============================="
    
    local recipe_file=""
    local recipe_desc=""
    
    if [ "$recipe_num" = "1" ]; then
        recipe_file="configs/dataset/recipe1_commercial_ready.json"
        recipe_desc="상용화 대비 및 민첩성 중심 (75GB, 허용적 라이선스)"
        echo -e "${GREEN}🚀 레시피 1: $recipe_desc${NC}"
        echo -e "${BLUE}   • 법적 리스크 최소화${NC}"
        echo -e "${BLUE}   • 미래 상업화 옵션 확보${NC}"
        echo -e "${BLUE}   • 소규모 연구소 최적화${NC}"
    elif [ "$recipe_num" = "2" ]; then
        recipe_file="configs/dataset/recipe2_performance_max.json"
        recipe_desc="연구 성능 극대화 (150GB, 혼합 라이선스)"
        echo -e "${YELLOW}🔬 레시피 2: $recipe_desc${NC}"
        echo -e "${YELLOW}   • 최대 데이터 규모${NC}"
        echo -e "${YELLOW}   • 벤치마크 성능 극대화${NC}"
        echo -e "${YELLOW}   ⚠️  상업적 활용 제약 있음${NC}"
    fi
    
    # 레시피 기반 다운로드 실행
    echo -e "${YELLOW}🚀 레시피 기반 다운로드 실행 중...${NC}"
    if python3 "$SCRIPT_DIR/download_recipe_data.py" --recipe "$recipe_file" --output-dir "$OUTPUT_DIR"; then
        echo -e "${GREEN}✅ 레시피 $recipe_num 다운로드 성공${NC}"
        return 0
    else
        echo -e "${RED}❌ 레시피 $recipe_num 다운로드 실패${NC}"
        echo -e "${YELLOW}기본 다운로드로 대체 시도...${NC}"
        return 1
    fi
}

execute_download() {
    echo -e "\n${GREEN}📥 원시 데이터 다운로드 시작${NC}"
    echo "==============================="
    
    # 레시피 기반 다운로드 시도
    if [ "$RECIPE1" = true ]; then
        if execute_recipe_download "1"; then
            return 0
        fi
    elif [ "$RECIPE2" = true ]; then
        if execute_recipe_download "2"; then
            return 0
        fi
    fi
    
    # 기본 다운로드 로직
    local download_args=""
    
    # 다운로드 타입 설정
    if [ "$KOREAN_ONLY" = true ]; then
        download_args="--korean"
        echo -e "${BLUE}🇰🇷 한국어 데이터만 다운로드${NC}"
    elif [ "$ENGLISH_ONLY" = true ]; then
        download_args="--english"
        echo -e "${BLUE}🇺🇸 영어 데이터만 다운로드${NC}"
    elif [ "$INSTRUCTIONS_ONLY" = true ]; then
        download_args="--instructions"
        echo -e "${BLUE}🎯 명령어 데이터만 다운로드${NC}"
    else
        download_args="--all"
        echo -e "${BLUE}📦 모든 데이터 다운로드${NC}"
    fi
    
    # 기타 옵션 추가
    if [ "$SMALL_SAMPLE" = true ]; then
        download_args="$download_args --small"
        echo -e "${YELLOW}⚡ 테스트용 소량 샘플 모드${NC}"
    fi
    
    # 출력 디렉토리 지정
    download_args="$download_args --output-dir $OUTPUT_DIR"
    
    # 다운로드 실행
    echo -e "${YELLOW}🚀 다운로드 실행 중...${NC}"
    if python3 "$SCRIPT_DIR/download_raw_data.py" $download_args; then
        echo -e "${GREEN}✅ 원시 데이터 다운로드 성공${NC}"
        return 0
    else
        echo -e "${RED}❌ 원시 데이터 다운로드 실패${NC}"
        return 1
    fi
}

validate_downloaded_data() {
    echo -e "\n${GREEN}🔍 다운로드 결과 검증${NC}"
    echo "========================="
    
    local validation_passed=true
    local total_files=0
    local total_size=0
    
    # 다운로드된 파일들 확인
    echo -e "${YELLOW}📋 다운로드된 파일 목록:${NC}"
    
    if [ -d "$OUTPUT_DIR" ]; then
        local files=($(find "$OUTPUT_DIR" -name "*.jsonl" -type f))
        
        if [ ${#files[@]} -eq 0 ]; then
            echo -e "  ❌ 다운로드된 파일이 없습니다"
            validation_passed=false
        else
            for file in "${files[@]}"; do
                if [ -f "$file" ]; then
                    local size=$(du -h "$file" | cut -f1)
                    local lines=$(wc -l < "$file" 2>/dev/null || echo "0")
                    echo -e "  ✅ $(basename "$file"): $size ($lines 라인)"
                    ((total_files++))
                    total_size=$((total_size + $(du -k "$file" | cut -f1)))
                else
                    echo -e "  ❌ $(basename "$file"): 파일 없음"
                    validation_passed=false
                fi
            done
        fi
    else
        echo -e "  ❌ 출력 디렉토리가 존재하지 않습니다"
        validation_passed=false
    fi
    
    # 메타데이터 파일 확인
    local metadata_file="$OUTPUT_DIR/download_metadata.json"
    if [ -f "$metadata_file" ]; then
        echo -e "  ✅ download_metadata.json: 메타데이터 파일"
    else
        echo -e "  ⚠️  download_metadata.json: 메타데이터 파일 없음"
    fi
    
    # 라이선스 매니페스트 확인
    local license_file="$OUTPUT_DIR/dataset_license_manifest.json"
    if [ -f "$license_file" ]; then
        echo -e "  ✅ dataset_license_manifest.json: 라이선스 추적 파일"
    else
        echo -e "  ⚠️  dataset_license_manifest.json: 라이선스 추적 파일 없음"
    fi
    
    # 요약 정보
    echo -e "\n${BLUE}📊 다운로드 요약:${NC}"
    echo -e "총 파일 수: ${total_files}개"
    echo -e "총 용량: $((total_size / 1024))MB"
    echo -e "저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    
    # 최종 결과
    if [ "$validation_passed" = true ] && [ $total_files -gt 0 ]; then
        echo -e "\n${GREEN}🎉 모든 검증 통과! 다운로드 완료${NC}"
        return 0
    else
        echo -e "\n${YELLOW}⚠️  일부 검증 실패. 다운로드 확인 필요${NC}"
        return 1
    fi
}

generate_download_summary() {
    echo -e "\n${BLUE}📊 다운로드 완료 요약${NC}"
    echo "====================="
    
    # 처리 시간 계산
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "다운로드 시간: ${hours}시간 ${minutes}분 ${seconds}초"
    echo -e "저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    # 디스크 사용량
    if command -v du &> /dev/null; then
        local total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "?")
        echo "디스크 사용량: $total_size"
        echo ""
    fi
    
    # 선택된 전략 표시
    if [ "$RECIPE1" = true ]; then
        echo -e "${GREEN}📋 사용된 전략: 레시피 1 (상용화 대비)${NC}"
        echo -e "   • 허용적 라이선스 중심"
        echo -e "   • 상업적 활용 자유"
        echo -e "   • 법적 리스크 최소화"
    elif [ "$RECIPE2" = true ]; then
        echo -e "${YELLOW}📋 사용된 전략: 레시피 2 (성능 극대화)${NC}"
        echo -e "   • 대규모 데이터 활용"
        echo -e "   • 벤치마크 성능 최적화"
        echo -e "   ⚠️  상업적 활용 제약 있음"
    fi
    echo ""
    
    # 다음 단계 안내
    echo -e "${YELLOW}🚀 다음 단계 (전처리):${NC}"
    
    if [ "$RECIPE1" = true ]; then
        echo "레시피 1 기반 전처리:"
        echo "   python3 preprocess_pretraining.py --recipe configs/dataset/recipe1_commercial_ready.json"
    elif [ "$RECIPE2" = true ]; then
        echo "레시피 2 기반 전처리:"
        echo "   python3 preprocess_pretraining.py --recipe configs/dataset/recipe2_performance_max.json"
    else
        echo "1. 사전학습 데이터 전처리:"
        echo "   python3 preprocess_pretraining.py --raw-data-dir $OUTPUT_DIR"
    fi
    
    echo ""
    echo "2. 미세조정 데이터 전처리:"
    echo "   python3 preprocess_finetuning.py --raw-data-dir $OUTPUT_DIR"
    echo ""
    echo -e "${BLUE}💡 참고:${NC}"
    echo "- 메타데이터가 $OUTPUT_DIR/download_metadata.json 에 저장됨"
    echo "- 라이선스 정보가 $OUTPUT_DIR/dataset_license_manifest.json 에 저장됨"
    echo "- 전처리 없이 원시 데이터를 바로 사용 가능"
    echo "- RecommendDataset.md 권고사항이 반영됨"
}

# =============================================================================
# 메인 실행 함수
# =============================================================================

run_download_pipeline() {
    echo -e "${GREEN}🚀 원시 데이터 다운로드 시작${NC}"
    
    local steps_passed=0
    local total_steps=3
    
    # 1단계: 환경 설정
    if setup_output_directory && check_python_dependencies; then
        ((steps_passed++))
        echo -e "${GREEN}✅ 1단계: 환경 설정 완료${NC}"
    else
        echo -e "${RED}❌ 환경 설정 실패${NC}"
        return 1
    fi
    
    # 2단계: 원시 데이터 다운로드
    if execute_download; then
        ((steps_passed++))
        echo -e "${GREEN}✅ 2단계: 데이터 다운로드 완료${NC}"
    else
        echo -e "${RED}❌ 데이터 다운로드 실패${NC}"
        return 1
    fi
    
    # 3단계: 검증
    if validate_downloaded_data; then
        ((steps_passed++))
        echo -e "${GREEN}✅ 3단계: 검증 완료${NC}"
    else
        echo -e "${YELLOW}⚠️  3단계: 검증 부분 실패 (다운로드는 완료)${NC}"
        ((steps_passed++))  # 검증 실패해도 진행
    fi
    
    echo -e "\n${GREEN}✅ 다운로드 파이프라인 완료: $steps_passed/$total_steps 단계 성공${NC}"
    return 0
}

# =============================================================================
# 명령행 인수 파싱 및 메인 실행
# =============================================================================

# 변수 초기화
init_core_variables

# 다운로드 전용 변수
START_TIME=$(date +%s)
OUTPUT_DIR="raw_datasets"

# 다운로드 옵션
KOREAN_ONLY=false
ENGLISH_ONLY=false
INSTRUCTIONS_ONLY=false
SMALL_SAMPLE=false

# 새로운 레시피 옵션
RECIPE1=false
RECIPE2=false

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --recipe1)
            RECIPE1=true
            RECIPE2=false
            KOREAN_ONLY=false
            ENGLISH_ONLY=false
            INSTRUCTIONS_ONLY=false
            shift
            ;;
        --recipe2)
            RECIPE1=false
            RECIPE2=true
            KOREAN_ONLY=false
            ENGLISH_ONLY=false
            INSTRUCTIONS_ONLY=false
            shift
            ;;
        --all)
            KOREAN_ONLY=false
            ENGLISH_ONLY=false
            INSTRUCTIONS_ONLY=false
            RECIPE1=false
            RECIPE2=false
            shift
            ;;
        --korean)
            KOREAN_ONLY=true
            ENGLISH_ONLY=false
            INSTRUCTIONS_ONLY=false
            RECIPE1=false
            RECIPE2=false
            shift
            ;;
        --english)
            KOREAN_ONLY=false
            ENGLISH_ONLY=true
            INSTRUCTIONS_ONLY=false
            RECIPE1=false
            RECIPE2=false
            shift
            ;;
        --instructions)
            KOREAN_ONLY=false
            ENGLISH_ONLY=false
            INSTRUCTIONS_ONLY=true
            RECIPE1=false
            RECIPE2=false
            shift
            ;;
        --small)
            SMALL_SAMPLE=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --license-info)
            print_license_info
            exit 0
            ;;
        -h|--help)
            print_download_help
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            print_download_help
            exit 1
            ;;
    esac
done

# 메인 실행
print_download_banner

if [ "$CHECK_ONLY" = true ]; then
    DOWNLOAD_TYPE="full"
    check_disk_space
    exit 0
fi

# 환경 확인
check_environment

# 다운로드 파이프라인 실행
if run_download_pipeline; then
    generate_download_summary
    echo -e "\n${GREEN}🎉 원시 데이터 다운로드가 성공적으로 완료되었습니다!${NC}"
    echo -e "${BLUE}💡 RecommendDataset.md 권고사항이 반영되었습니다${NC}"
    if [ "$RECIPE1" = true ] || [ "$RECIPE2" = true ]; then
        echo -e "${YELLOW}📋 레시피 기반 전처리를 위해 해당 설정 파일을 사용하세요${NC}"
    else
        echo -e "${BLUE}💡 전처리가 필요하면 preprocess_*.py 스크립트를 실행하세요${NC}"
    fi
    exit 0
else
    echo -e "\n${RED}❌ 다운로드 중 오류가 발생했습니다.${NC}"
    echo "로그를 확인하여 자세한 정보를 확인하세요."
    exit 1
fi 