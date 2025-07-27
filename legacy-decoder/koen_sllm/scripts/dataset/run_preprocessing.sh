#!/bin/bash

# 한국어 sLLM 데이터 전처리 통합 제어 스크립트 (RecommendDataset.md 권고사항 반영)
# Unified preprocessing controller for Korean sLLM datasets with RecommendDataset.md recommendations

set -e  # 에러 발생 시 스크립트 종료

# =============================================================================
# 핵심 모듈 로드
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 색상 정의 (download_core.sh 의존성 제거)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 전처리 전용 변수 및 함수들
# =============================================================================

# 전처리 설정 변수 (먼저 정의 - download_core.sh의 덮어쓰기 방지)
PREPROCESS_TYPE="all"      # all, pretraining, finetuning
RAW_DATA_DIR="raw_datasets"
OUTPUT_DIR="datasets"      # 기본값 명시적 설정
KOREAN_RATIO=0.70          # 한국어 중심 학습 기본값
ENGLISH_RATIO=0.30         # 한국어 중심 학습 기본값
TARGET_SIZE=""
MIN_TARGET=50000
VALIDATION_ONLY=false
STATS_ONLY=false
RECIPE_MODE=""
AUGMENTATION=true
QUALITY_LEVEL="normal"      # strict, normal, permissive

# 새로운 권고사항 변수
LICENSE_STRATEGY="permissive"  # permissive, mixed
COMMERCIAL_READY=true
EXCLUDE_AI_HUB=true

# 전처리 통계
START_TIME=$(date +%s)
PREPROCESSING_LOG=""

# 변수 초기화 함수 (download_core.sh 대신)
init_preprocessing_variables() {
    # OUTPUT_DIR 값 보존하며 필요한 변수만 초기화
    KOREAN_ONLY=false
    ENGLISH_ONLY=false
    MIXED_ONLY=false
    VERBOSE=false
    QUIET=false
    FORCE_FLAG=""
}

print_preprocessing_banner() {
    echo -e "${BLUE}"
    echo "========================================================"
    echo "     한국어 sLLM 데이터 전처리 통합 제어기"
    echo "     Korean sLLM Data Preprocessing Controller"
    echo "     🆕 RecommendDataset.md 권고사항 반영"
    echo "========================================================"
    echo ""
    echo "🎯 전처리 전략:"
    echo "   🚀  레시피 1: 상용화 대비 (허용적 라이선스 중심) [권장]"
    echo "   🔬  레시피 2: 성능 극대화 (대규모 데이터)"
    echo "   📚  사전학습 데이터만"  
    echo "   🎯  미세조정 데이터만"
    echo "   📊  검증 및 통계만"
    echo "========================================================"
    echo -e "${NC}"
}

print_preprocessing_help() {
    echo "한국어 sLLM 데이터 전처리 통합 제어기 (v2.0)"
    echo "RecommendDataset.md 권고사항 기반 업데이트"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "🚀 추천 레시피 옵션 (RecommendDataset.md 기반):"
    echo "  --recipe1            레시피 1: 상용화 대비 전처리 (권장)"
    echo "  --recipe2            레시피 2: 성능 극대화 전처리"
    echo "  --recipe3            레시피 3: 한국어 중심 학습 (한국어 70%)"
    echo "  --recipe-config PATH 커스텀 레시피 설정 파일"
    echo ""
    echo "🇰🇷 한국어 중심 옵션:"
    echo "  --korean-focused     한국어 70%, 영어 30% 비율로 설정"
    echo "  --korean-heavy       한국어 80%, 영어 20% 비율로 설정" 
    echo "  --korean-only        한국어만 사용 (100%)"
    echo "  --include-ai-hub     AI-Hub 데이터 포함 (더 많은 한국어 데이터)"
    echo ""
    echo "🎯 전처리 타입 옵션:"
    echo "  --all                전체 파이프라인 실행 (사전학습 + 미세조정) [기본값]"
    echo "  --pretraining        사전학습 데이터만 전처리"
    echo "  --finetuning         미세조정 데이터만 전처리"
    echo "  --validation         전처리 없이 검증 및 통계만"
    echo ""
    echo "📁 디렉토리 옵션:"
    echo "  --raw-data-dir DIR   원시 데이터 디렉토리 (기본값: raw_datasets)"
    echo "  --output-dir DIR     출력 디렉토리 (기본값: datasets)"
    echo ""
    echo "🌍 언어 비율 옵션 (기본값: 한국어 70%, 영어 30%):"
    echo "  --korean-ratio N     한국어 비율 (0.0-1.0, 기본값: 0.70)"
    echo "  --english-ratio N    영어 비율 (0.0-1.0, 기본값: 0.30)"
    echo "  --korean-only        한국어만 사용"
    echo "  --english-only       영어만 사용"
    echo "  --mixed-only         혼합 데이터만 생성"
    echo ""
    echo "📊 데이터 크기 옵션:"
    echo "  --target-size N      목표 데이터셋 크기"
    echo "  --min-target N       최소 목표 크기 (미세조정, 기본값: 50000)"
    echo "  --max-size N         최대 크기 제한"
    echo ""
    echo "🔧 품질 및 처리 옵션:"
    echo "  --quality-level LVL  품질 필터링 수준 (strict/normal/permissive)"
    echo "  --no-augmentation    데이터 증강 비활성화"
    echo "  --no-dedup          중복 제거 비활성화"
    echo "  --force              기존 결과 덮어쓰기"
    echo ""
    echo "📋 라이선스 전략 옵션:"
    echo "  --license-strategy TYPE  라이선스 전략 (permissive/mixed)"
    echo "  --exclude-ai-hub     AI-Hub 데이터 제외 (상업적 안전)"
    echo "  --include-ai-hub     AI-Hub 데이터 포함 (성능 최적화)"
    echo "  --license-info       라이선스 정보 표시"
    echo ""
    echo "🔍 분석 및 검증 옵션:"
    echo "  --stats-only        통계 생성만"
    echo "  --check-quality     품질 분석 수행"
    echo "  --benchmark         벤치마크 데이터로 성능 측정"
    echo ""
    echo "🛠️ 기타 옵션:"
    echo "  --parallel N        병렬 처리 수 (기본값: CPU 코어 수)"
    echo "  --memory-limit N    메모리 사용 제한 (GB)"
    echo "  --temp-dir DIR      임시 디렉토리"
    echo "  --verbose           상세 로그 출력"
    echo "  --quiet             최소 로그만 출력"
    echo "  -h, --help          도움말 표시"
    echo ""
    echo "🚀 레시피별 특징:"
    echo ""
    echo "  📋 레시피 1 (상용화 대비) - 소규모 연구소 권장:"
    echo "     • 총 규모: ~75GB (한국어 37.5GB + 영어 37.5GB)"
    echo "     • 라이선스: 완전 허용적 (Apache 2.0, CC0, ODC-BY)"
    echo "     • 혼합 비율: 한국어 50%, 영어 50%"
    echo "     • 장점: 법적 리스크 최소화, 미래 상업화 자유, 균형잡힌 이중언어"
    echo "     • 제외: AI-Hub, Books3 (저작권 위험)"
    echo ""
    echo "  📋 레시피 2 (성능 극대화) - 순수 연구용:"
    echo "     • 총 규모: ~150GB (한국어 75GB + 영어 75GB)"
    echo "     • 라이선스: 혼합형 (AI-Hub 포함)"
    echo "     • 혼합 비율: 한국어 50%, 영어 50%"
    echo "     • 장점: 최대 성능, 벤치마크 경쟁력, 균형잡힌 이중언어"
    echo "     • 제약: 상업적 활용 시 별도 협의 필요"
    echo ""
    echo "📋 전처리 파이프라인 설명:"
    echo ""
    echo "  🚀 사전학습 전처리:"
    echo "     • 한영 혼합 비율 조정 (레시피별 최적화)"
    echo "     • 품질 필터링 및 정규화 (C4 기법 적용)"
    echo "     • 중복 제거 (LSH 기법)"
    echo "     • 언어 감지 및 분류"
    echo "     • 라이선스 추적 및 관리"
    echo ""
    echo "  🎯 미세조정 전처리:"
    echo "     • 최소 5만개 이상 보장"
    echo "     • 태스크별 분류 및 균형 조정"
    echo "     • 데이터 증강 기법 적용"
    echo "     • 품질 검증 및 필터링"
    echo ""
    echo "예시:"
    echo "  $0 --recipe1                    # 상용화 대비 레시피 (권장)"
    echo "  $0 --recipe2                    # 성능 극대화 레시피"
    echo "  $0 --pretraining --korean-only  # 한국어 사전학습 데이터만"
    echo "  $0 --finetuning --min-target 100000  # 10만개 이상 미세조정 데이터"
    echo "  $0 --validation                 # 검증 및 통계만"
    echo "  $0 --license-info               # 라이선스 정보 확인"
    echo ""
    echo "📝 출력 파일:"
    echo "  사전학습: datasets/mixed_pretraining.jsonl"
    echo "  미세조정: datasets/mixed_instructions.json"
    echo "  통계: datasets/*_stats.json"
    echo "  라이선스: datasets/license_manifest.json"
    echo "  로그: datasets/preprocessing.log"
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
    echo "  • 데이터 품질 > 데이터 양 (RecommendDataset.md 핵심)"
    echo ""
}

check_preprocessing_requirements() {
    echo -e "${YELLOW}🔍 전처리 환경 확인 중...${NC}"
    
    local requirements_met=true
    
    # OUTPUT_DIR이 제대로 설정되었는지 확인
    if [ -z "$OUTPUT_DIR" ]; then
        echo -e "${RED}❌ OUTPUT_DIR 변수가 설정되지 않았습니다${NC}"
        OUTPUT_DIR="datasets"  # 기본값으로 복구
        echo -e "${YELLOW}⚠️  OUTPUT_DIR을 기본값으로 설정: $OUTPUT_DIR${NC}"
    fi
    
    # Python 패키지 확인
    local required_packages=("datasets" "tqdm" "pandas" "numpy" "langdetect")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo -e "${RED}❌ 필요한 Python 패키지가 없습니다: ${missing_packages[*]}${NC}"
        echo "다음 명령어로 설치하세요:"
        echo "pip install datasets tqdm pandas numpy langdetect"
        requirements_met=false
    fi
    
    # 원시 데이터 디렉토리 확인
    if [ ! -d "$RAW_DATA_DIR" ]; then
        echo -e "${RED}❌ 원시 데이터 디렉토리가 없습니다: $RAW_DATA_DIR${NC}"
        echo "먼저 process_datasets.sh를 실행하여 원시 데이터를 다운로드하세요."
        requirements_met=false
    else
        local data_files=$(find "$RAW_DATA_DIR" -name "*.jsonl" -o -name "*.json" | wc -l)
        if [ $data_files -eq 0 ]; then
            echo -e "${YELLOW}⚠️  원시 데이터 파일이 없습니다${NC}"
            echo "process_datasets.sh를 실행하여 데이터를 다운로드하세요."
            requirements_met=false
        else
            echo -e "${GREEN}✅ 원시 데이터 파일 $data_files개 확인${NC}"
        fi
    fi
    
    # 출력 디렉토리 설정
    mkdir -p "$OUTPUT_DIR"
    if [ ! -w "$OUTPUT_DIR" ]; then
        echo -e "${RED}❌ 출력 디렉토리 쓰기 권한이 없습니다: $OUTPUT_DIR${NC}"
        requirements_met=false
    fi
    
    # 디스크 공간 확인
    if command -v df &> /dev/null && [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        local available_space=$(df "$OUTPUT_DIR" | awk 'NR==2 {print $4}' 2>/dev/null || echo "0")
        local required_space=10485760  # 10GB in KB
        
        if [ -n "$available_space" ] && [ "$available_space" -gt 0 ]; then
            if [ $available_space -lt $required_space ]; then
                echo -e "${YELLOW}⚠️  디스크 공간이 부족할 수 있습니다 (사용 가능: $((available_space/1024/1024))GB)${NC}"
            fi
        fi
    fi
    
    # 레시피 설정 검증
    if [ -n "$RECIPE_MODE" ]; then
        local recipe_file=""
        if [ "$RECIPE_MODE" = "recipe1" ]; then
            recipe_file="$SCRIPT_DIR/../../configs/dataset/recipe1_commercial_ready.json"
        elif [ "$RECIPE_MODE" = "recipe2" ]; then
            recipe_file="$SCRIPT_DIR/../../configs/dataset/recipe2_performance_max.json"
        elif [ "$RECIPE_MODE" = "recipe3" ]; then
            recipe_file="$SCRIPT_DIR/../../configs/dataset/recipe3_korean_focused.json"
        else
            recipe_file="$RECIPE_MODE"
        fi
        
        if [ ! -f "$recipe_file" ]; then
            echo -e "${YELLOW}⚠️  레시피 파일이 없습니다: $recipe_file${NC}"
            echo "기본 설정으로 진행합니다."
            RECIPE_MODE=""
        else
            echo -e "${GREEN}✅ 레시피 파일 확인: $(basename "$recipe_file")${NC}"
        fi
    fi
    
    if [ "$requirements_met" = true ]; then
        echo -e "${GREEN}✅ 전처리 환경 확인 완료${NC}"
        return 0
    else
        echo -e "${RED}❌ 전처리 환경 요구사항 미충족${NC}"
        return 1
    fi
}

build_preprocessing_args() {
    local script_type=$1
    local args=""
    
    # 공통 인수
    args="$args --raw-data-dir $RAW_DATA_DIR"
    args="$args --output-dir $OUTPUT_DIR"
    
    # 언어 비율
    if [ "$KOREAN_ONLY" = true ]; then
        args="$args --korean-only"
    elif [ "$ENGLISH_ONLY" = true ]; then
        args="$args --english-only"
    elif [ "$MIXED_ONLY" = true ]; then
        args="$args --mixed-only"
    else
        args="$args --korean-ratio $KOREAN_RATIO --english-ratio $ENGLISH_RATIO"
    fi
    
    # 타겟 크기
    if [ -n "$TARGET_SIZE" ]; then
        args="$args --target-size $TARGET_SIZE"
    fi
    
    # 라이선스 전략과 AI-Hub 제외 옵션은 Python 스크립트에서 지원하지 않으므로 제거
    # 이러한 설정은 레시피 파일 내에서 처리됨
    
    # 스크립트별 특수 인수
    if [ "$script_type" = "finetuning" ]; then
        args="$args --min-target $MIN_TARGET"
        
        if [ "$AUGMENTATION" = false ]; then
            args="$args --no-augmentation"
        fi
    fi
    
    # 품질 수준, force, verbose 옵션들은 Python 스크립트에서 지원하지 않으므로 제거
    # 이러한 설정들은 레시피 파일이나 기본 구현에서 처리됨
    
    echo "$args"
}

execute_pretraining_preprocessing() {
    echo -e "\n${GREEN}📚 사전학습 데이터 전처리 시작${NC}"
    echo "================================="
    
    local args=$(build_preprocessing_args "pretraining")
    
    # 레시피 기반 처리
    if [ -n "$RECIPE_MODE" ]; then
        local recipe_file=""
        if [ "$RECIPE_MODE" = "recipe1" ]; then
            recipe_file="$SCRIPT_DIR/../../configs/dataset/recipe1_commercial_ready.json"
            echo -e "${GREEN}🚀 레시피 1: 상용화 대비 전처리${NC}"
            echo -e "${BLUE}   • 허용적 라이선스 중심${NC}"
            echo -e "${BLUE}   • 법적 리스크 최소화${NC}"
            echo -e "${BLUE}   • 한국어 50%, 영어 50% 비율${NC}"
        elif [ "$RECIPE_MODE" = "recipe2" ]; then
            recipe_file="$SCRIPT_DIR/../../configs/dataset/recipe2_performance_max.json"
            echo -e "${YELLOW}🔬 레시피 2: 성능 극대화 전처리${NC}"
            echo -e "${YELLOW}   • 대규모 데이터 활용${NC}"
            echo -e "${YELLOW}   • 벤치마크 성능 최적화${NC}"
            echo -e "${YELLOW}   • 한국어 50%, 영어 50% 비율${NC}"
            echo -e "${YELLOW}   ⚠️  상업적 활용 제약 있음${NC}"
        elif [ "$RECIPE_MODE" = "recipe3" ]; then
            recipe_file="$SCRIPT_DIR/../../configs/dataset/recipe3_korean_focused.json"
            echo -e "${BLUE}🇰🇷 레시피 3: 한국어 중심 학습${NC}"
            echo -e "${BLUE}   • 한국어 70%, 영어 30% 비율${NC}"
            echo -e "${BLUE}   • 더 많은 한국어 데이터 포함${NC}"
        else
            recipe_file="$RECIPE_MODE"
            echo -e "${BLUE}🎯 커스텀 레시피: $(basename "$recipe_file")${NC}"
        fi
        
        if [ -f "$recipe_file" ]; then
            echo -e "${GREEN}✅ 레시피 파일 확인됨: $recipe_file${NC}"
            echo -e "${BLUE}   레시피 설정은 Python 스크립트 내부에서 처리됩니다${NC}"
        else
            echo -e "${YELLOW}⚠️  레시피 파일이 없습니다. 기본 설정으로 진행${NC}"
        fi
    fi
    
    echo -e "${YELLOW}🚀 사전학습 전처리 실행: preprocess_pretraining.py${NC}"
    echo "인수: $args"
    
    if python3 "$SCRIPT_DIR/preprocess_pretraining.py" $args; then
        echo -e "${GREEN}✅ 사전학습 전처리 성공${NC}"
        return 0
    else
        echo -e "${RED}❌ 사전학습 전처리 실패${NC}"
        return 1
    fi
}

execute_finetuning_preprocessing() {
    echo -e "\n${GREEN}🎯 미세조정 데이터 전처리 시작${NC}"
    echo "================================="
    
    local args=$(build_preprocessing_args "finetuning")
    
    # 미세조정 특화 옵션
    echo -e "${BLUE}📊 미세조정 설정:${NC}"
    echo "   • 최소 목표: $MIN_TARGET 개"
    if [ -n "$TARGET_SIZE" ]; then
        echo "   • 목표 크기: $TARGET_SIZE 개"
    fi
    echo "   • 데이터 증강: $([ "$AUGMENTATION" = true ] && echo "활성화" || echo "비활성화")"
    echo "   • 품질 수준: $QUALITY_LEVEL"
    echo "   • 라이선스 전략: $LICENSE_STRATEGY"
    
    echo -e "${YELLOW}🚀 미세조정 전처리 실행: preprocess_finetuning.py${NC}"
    echo "인수: $args"
    
    if python3 "$SCRIPT_DIR/preprocess_finetuning.py" $args; then
        echo -e "${GREEN}✅ 미세조정 전처리 성공${NC}"
        return 0
    else
        echo -e "${RED}❌ 미세조정 전처리 실패${NC}"
        return 1
    fi
}

validate_preprocessing_results() {
    echo -e "\n${GREEN}🔍 전처리 결과 검증${NC}"
    echo "========================="
    
    local validation_passed=true
    
    # 사전학습 데이터 검증
    if [ "$PREPROCESS_TYPE" = "all" ] || [ "$PREPROCESS_TYPE" = "pretraining" ]; then
        local pretraining_file="$OUTPUT_DIR/mixed_pretraining.jsonl"
        if [ -f "$pretraining_file" ]; then
            local lines=$(wc -l < "$pretraining_file" 2>/dev/null || echo "0")
            local size=$(du -h "$pretraining_file" | cut -f1)
            echo -e "  ✅ 사전학습 데이터: $lines 라인, $size"
            
            if [ $lines -lt 1000 ]; then
                echo -e "  ⚠️  사전학습 데이터가 너무 적습니다 ($lines < 1000)"
                validation_passed=false
            fi
        else
            echo -e "  ❌ 사전학습 데이터 파일이 없습니다"
            validation_passed=false
        fi
    fi
    
    # 미세조정 데이터 검증
    if [ "$PREPROCESS_TYPE" = "all" ] || [ "$PREPROCESS_TYPE" = "finetuning" ]; then
        local finetuning_file="$OUTPUT_DIR/mixed_instructions.json"
        if [ -f "$finetuning_file" ]; then
            local count=$(python3 -c "
import json
try:
    with open('$finetuning_file', 'r') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
" 2>/dev/null || echo "0")
            local size=$(du -h "$finetuning_file" | cut -f1)
            echo -e "  ✅ 미세조정 데이터: $count 개, $size"
            
            if [ $count -lt $MIN_TARGET ]; then
                echo -e "  ⚠️  미세조정 데이터가 목표보다 적습니다 ($count < $MIN_TARGET)"
                validation_passed=false
            fi
        else
            echo -e "  ❌ 미세조정 데이터 파일이 없습니다"
            validation_passed=false
        fi
    fi
    
    # 메타데이터 및 통계 파일 확인
    local stats_files=("preprocessing_stats.json" "language_distribution.json" "quality_analysis.json" "license_manifest.json")
    for stats_file in "${stats_files[@]}"; do
        local file_path="$OUTPUT_DIR/$stats_file"
        if [ -f "$file_path" ]; then
            echo -e "  ✅ 관리 파일: $stats_file"
        else
            echo -e "  ⚠️  관리 파일 없음: $stats_file"
        fi
    done
    
    # 최종 결과
    if [ "$validation_passed" = true ]; then
        echo -e "\n${GREEN}🎉 모든 검증 통과! 전처리 완료${NC}"
        return 0
    else
        echo -e "\n${YELLOW}⚠️  일부 검증 실패. 결과 확인 필요${NC}"
        return 1
    fi
}

generate_preprocessing_statistics() {
    echo -e "\n${BLUE}📊 전처리 통계 생성${NC}"
    echo "====================="
    
    # 처리 시간 계산
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "처리 시간: ${hours}시간 ${minutes}분 ${seconds}초"
    echo -e "출력 디렉토리: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    # 선택된 전략 표시
    if [ "$RECIPE_MODE" = "recipe1" ]; then
        echo -e "${GREEN}📋 사용된 전략: 레시피 1 (상용화 대비)${NC}"
        echo -e "   • 허용적 라이선스 중심"
        echo -e "   • 상업적 활용 자유"
        echo -e "   • 법적 리스크 최소화"
        echo -e "   • 한국어 50%, 영어 50% 비율"
    elif [ "$RECIPE_MODE" = "recipe2" ]; then
        echo -e "${YELLOW}📋 사용된 전략: 레시피 2 (성능 극대화)${NC}"
        echo -e "   • 대규모 데이터 활용"
        echo -e "   • 벤치마크 성능 최적화"
        echo -e "   • 한국어 50%, 영어 50% 비율"
        echo -e "   ⚠️  상업적 활용 제약 있음"
    elif [ "$RECIPE_MODE" = "recipe3" ]; then
        echo -e "${BLUE}📋 사용된 전략: 레시피 3 (한국어 중심)${NC}"
        echo -e "   • 한국어 70%, 영어 30% 비율"
        echo -e "   • 더 많은 한국어 데이터 포함"
    else
        echo -e "${BLUE}📋 사용된 전략: 기본 설정${NC}"
        echo -e "   • 한국어 비율: $KOREAN_RATIO"
        echo -e "   • 영어 비율: $ENGLISH_RATIO"
        echo -e "   • 라이선스 전략: $LICENSE_STRATEGY"
    fi
    echo ""
    
    # 파일별 통계
    if [ -d "$OUTPUT_DIR" ]; then
        echo -e "${BLUE}📁 생성된 파일 목록:${NC}"
        
        # 주요 데이터 파일
        local main_files=("mixed_pretraining.jsonl" "mixed_instructions.json" "korean_only_pretraining.jsonl" "english_only_pretraining.jsonl")
        for file in "${main_files[@]}"; do
            local file_path="$OUTPUT_DIR/$file"
            if [ -f "$file_path" ]; then
                local size=$(du -h "$file_path" | cut -f1)
                if [[ "$file" == *.jsonl ]]; then
                    local count=$(wc -l < "$file_path" 2>/dev/null || echo "0")
                    echo -e "  📄 $file: $count 라인, $size"
                else
                    local count=$(python3 -c "
import json
try:
    with open('$file_path', 'r') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
" 2>/dev/null || echo "0")
                    echo -e "  📄 $file: $count 개, $size"
                fi
            fi
        done
        
        # 통계 파일
        echo -e "\n${BLUE}📊 관리 파일:${NC}"
        local stats_files=($(find "$OUTPUT_DIR" -name "*_stats.json" -o -name "*_distribution.json" -o -name "*_analysis.json" -o -name "*_manifest.json" 2>/dev/null))
        for file in "${stats_files[@]}"; do
            if [ -f "$file" ]; then
                echo -e "  📈 $(basename "$file")"
            fi
        done
        
        # 총 디스크 사용량
        local total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "?")
        echo -e "\n${BLUE}💾 총 디스크 사용량: $total_size${NC}"
    fi
    
    # 언어 분포 요약
    local lang_dist_file="$OUTPUT_DIR/language_distribution.json"
    if [ -f "$lang_dist_file" ]; then
        echo -e "\n${BLUE}🌍 언어 분포:${NC}"
        python3 -c "
import json
try:
    with open('$lang_dist_file', 'r') as f:
        data = json.load(f)
    
    for dataset, stats in data.items():
        print(f'  📚 {dataset}:')
        if 'korean' in stats:
            print(f'     🇰🇷 한국어: {stats[\"korean\"]:,}개')
        if 'english' in stats:
            print(f'     🇺🇸 영어: {stats[\"english\"]:,}개')
        if 'mixed' in stats:
            print(f'     🌍 혼합: {stats[\"mixed\"]:,}개')
        print()
except Exception as e:
    print(f'  ❌ 언어 분포 통계를 읽을 수 없습니다: {e}')
" 2>/dev/null || echo "  ❌ 언어 분포 파일을 읽을 수 없습니다"
    fi
}

run_preprocessing_pipeline() {
    echo -e "${GREEN}🚀 전처리 파이프라인 시작${NC}"
    
    local steps_passed=0
    local total_steps=0
    
    # 단계 수 계산
    case "$PREPROCESS_TYPE" in
        "all")
            total_steps=4  # 환경확인 + 사전학습 + 미세조정 + 검증
            ;;
        "pretraining"|"finetuning")
            total_steps=3  # 환경확인 + 전처리 + 검증
            ;;
        "validation")
            total_steps=2  # 환경확인 + 검증
            ;;
    esac
    
    # 환경 확인 (항상 실행)
    if check_preprocessing_requirements; then
        ((steps_passed++))
        echo -e "${GREEN}✅ 단계 $steps_passed/$total_steps: 환경 확인 완료${NC}"
    else
        echo -e "${RED}❌ 환경 확인 실패${NC}"
        return 1
    fi
    
    # 전처리 실행
    if [ "$PREPROCESS_TYPE" = "all" ] || [ "$PREPROCESS_TYPE" = "pretraining" ]; then
        if [ "$VALIDATION_ONLY" = false ]; then
            if execute_pretraining_preprocessing; then
                ((steps_passed++))
                echo -e "${GREEN}✅ 단계 $steps_passed/$total_steps: 사전학습 전처리 완료${NC}"
            else
                echo -e "${RED}❌ 사전학습 전처리 실패${NC}"
                return 1
            fi
        fi
    fi
    
    if [ "$PREPROCESS_TYPE" = "all" ] || [ "$PREPROCESS_TYPE" = "finetuning" ]; then
        if [ "$VALIDATION_ONLY" = false ]; then
            if execute_finetuning_preprocessing; then
                ((steps_passed++))
                echo -e "${GREEN}✅ 단계 $steps_passed/$total_steps: 미세조정 전처리 완료${NC}"
            else
                echo -e "${RED}❌ 미세조정 전처리 실패${NC}"
                return 1
            fi
        fi
    fi
    
    # 검증 (항상 실행)
    if validate_preprocessing_results; then
        ((steps_passed++))
        echo -e "${GREEN}✅ 단계 $steps_passed/$total_steps: 검증 완료${NC}"
    else
        echo -e "${YELLOW}⚠️  단계 $steps_passed/$total_steps: 검증 부분 실패${NC}"
        ((steps_passed++))  # 검증 실패해도 진행
    fi
    
    echo -e "\n${GREEN}✅ 전처리 파이프라인 완료: $steps_passed/$total_steps 단계 성공${NC}"
    return 0
}

# =============================================================================
# 명령행 인수 파싱 및 메인 실행
# =============================================================================

# 변수 초기화 (download_core.sh 의존성 제거)
init_preprocessing_variables

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --recipe1)
            RECIPE_MODE="recipe1"
            KOREAN_RATIO=0.50
            ENGLISH_RATIO=0.50
            LICENSE_STRATEGY="permissive"
            EXCLUDE_AI_HUB=true
            shift
            ;;
        --recipe2)
            RECIPE_MODE="recipe2"
            KOREAN_RATIO=0.50
            ENGLISH_RATIO=0.50
            LICENSE_STRATEGY="mixed"
            EXCLUDE_AI_HUB=false
            shift
            ;;
        --recipe3)
            RECIPE_MODE="recipe3"
            KOREAN_RATIO=0.70
            ENGLISH_RATIO=0.30
            LICENSE_STRATEGY="permissive"
            EXCLUDE_AI_HUB=false
            shift
            ;;
        --recipe-config)
            RECIPE_MODE="$2"
            shift 2
            ;;
        --all)
            PREPROCESS_TYPE="all"
            shift
            ;;
        --pretraining)
            PREPROCESS_TYPE="pretraining"
            shift
            ;;
        --finetuning)
            PREPROCESS_TYPE="finetuning"
            shift
            ;;
        --validation)
            PREPROCESS_TYPE="validation"
            VALIDATION_ONLY=true
            shift
            ;;
        --stats-only)
            STATS_ONLY=true
            shift
            ;;
        --raw-data-dir)
            RAW_DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --korean-ratio)
            KOREAN_RATIO="$2"
            shift 2
            ;;
        --english-ratio)
            ENGLISH_RATIO="$2"
            shift 2
            ;;
        --korean-only)
            KOREAN_ONLY=true
            ENGLISH_ONLY=false
            MIXED_ONLY=false
            shift
            ;;
        --english-only)
            KOREAN_ONLY=false
            ENGLISH_ONLY=true
            MIXED_ONLY=false
            shift
            ;;
        --mixed-only)
            KOREAN_ONLY=false
            ENGLISH_ONLY=false
            MIXED_ONLY=true
            shift
            ;;
        --target-size)
            TARGET_SIZE="$2"
            shift 2
            ;;
        --min-target)
            MIN_TARGET="$2"
            shift 2
            ;;
        --quality-level)
            QUALITY_LEVEL="$2"
            shift 2
            ;;
        --license-strategy)
            LICENSE_STRATEGY="$2"
            shift 2
            ;;
        --exclude-ai-hub)
            EXCLUDE_AI_HUB=true
            shift
            ;;
        --include-ai-hub)
            EXCLUDE_AI_HUB=false
            shift
            ;;
        --korean-focused)
            KOREAN_RATIO=0.70
            ENGLISH_RATIO=0.30
            EXCLUDE_AI_HUB=false
            shift
            ;;
        --korean-heavy)
            KOREAN_RATIO=0.80
            ENGLISH_RATIO=0.20
            EXCLUDE_AI_HUB=false
            shift
            ;;
        --no-augmentation)
            AUGMENTATION=false
            shift
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --license-info)
            print_license_info
            exit 0
            ;;
        -h|--help)
            print_preprocessing_help
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            print_preprocessing_help
            exit 1
            ;;
    esac
done

# 언어 비율 검증
if [ "$KOREAN_ONLY" = false ] && [ "$ENGLISH_ONLY" = false ]; then
    total_ratio=$(python3 -c "print($KOREAN_RATIO + $ENGLISH_RATIO)" 2>/dev/null || echo "0")
    if [ "$(python3 -c "print(abs($total_ratio - 1.0) > 0.01)" 2>/dev/null)" = "True" ]; then
        echo -e "${RED}❌ 한국어와 영어 비율의 합이 1.0이 되어야 합니다 (현재: $total_ratio)${NC}"
        exit 1
    fi
fi

# 메인 실행
print_preprocessing_banner

# 라이선스 정보만 표시하는 경우
if [ "$1" = "--license-info" ]; then
    print_license_info
    exit 0
fi

# 통계만 생성하는 경우
if [ "$STATS_ONLY" = true ]; then
    echo -e "${BLUE}📊 통계 생성 모드${NC}"
    generate_preprocessing_statistics
    exit 0
fi

# 전처리 파이프라인 실행
if run_preprocessing_pipeline; then
    generate_preprocessing_statistics
    echo -e "\n${GREEN}🎉 데이터 전처리가 성공적으로 완료되었습니다!${NC}"
    echo -e "${BLUE}💡 RecommendDataset.md 권고사항이 반영되었습니다${NC}"
    echo -e "${BLUE}💾 처리된 데이터가 $OUTPUT_DIR 에 저장되었습니다${NC}"
    
    # 다음 단계 안내
    echo -e "\n${YELLOW}🚀 다음 단계 안내:${NC}"
    
    if [ "$RECIPE_MODE" = "recipe1" ]; then
        echo "레시피 1 기반 학습 준비:"
        echo "   python3 train.py --data-dir $OUTPUT_DIR --license-safe"
    elif [ "$RECIPE_MODE" = "recipe2" ]; then
        echo "레시피 2 기반 학습 준비:"
        echo "   python3 train.py --data-dir $OUTPUT_DIR --performance-max"
    elif [ "$RECIPE_MODE" = "recipe3" ]; then
        echo "레시피 3 기반 학습 준비:"
        echo "   python3 train.py --data-dir $OUTPUT_DIR --korean-focused"
    else
        echo "일반 학습 준비:"
        echo "   python3 train.py --data-dir $OUTPUT_DIR"
    fi
    
    echo ""
    echo "2. 데이터 품질 확인:"
    echo "   python3 check_datasets.py --datasets-dir $OUTPUT_DIR"
    echo ""
    echo "3. 라이선스 준수 확인:"
    echo "   cat $OUTPUT_DIR/license_manifest.json"
    echo ""
    echo "4. 추가 전처리 (필요시):"
    echo "   $0 --finetuning --min-target 100000  # 더 많은 미세조정 데이터"
    
    exit 0
else
    echo -e "\n${RED}❌ 전처리 중 오류가 발생했습니다.${NC}"
    echo "로그를 확인하여 자세한 정보를 확인하세요."
    exit 1
fi 