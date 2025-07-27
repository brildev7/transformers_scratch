#!/bin/bash

# 한국어 sLLM 데이터셋 다운로드 스크립트
# Dataset download script for Korean sLLM

set -e  # 에러 발생 시 스크립트 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_banner() {
    echo -e "${BLUE}"
    echo "========================================================"
    echo "     한국어 sLLM 데이터셋 다운로드 스크립트"
    echo "     Korean sLLM Dataset Download Script"
    echo "========================================================"
    echo -e "${NC}"
}

print_help() {
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
    echo "예시:"
    echo "  $0               # 대화형 메뉴 시작"
    echo "  $0 --auto        # 모든 데이터셋 자동 다운로드"
    echo "  $0 --pretraining # 사전훈련용 데이터셋만"
    echo "  $0 --force       # 강제 새로 다운로드"
}

# JSON 파일에서 데이터셋 정보 읽기
get_dataset_info() {
    local config_file="$1"
    local info_type="$2"  # "names" 또는 "descriptions"
    
    if [ ! -f "$config_file" ]; then
        echo "설정 파일을 찾을 수 없습니다: $config_file" >&2
        return 1
    fi
    
    if [ "$info_type" = "names" ]; then
        python3 -c "
import json, sys
try:
    with open('$config_file', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for dataset in data.get('datasets', []):
        if dataset.get('enabled', True):
            print(dataset['name'])
except Exception as e:
    sys.exit(1)
"
    elif [ "$info_type" = "descriptions" ]; then
        python3 -c "
import json, sys
try:
    with open('$config_file', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i, dataset in enumerate(data.get('datasets', []), 1):
        if dataset.get('enabled', True):
            name = dataset['name']
            desc = dataset.get('description', name)
            priority = dataset.get('priority', 999)
            print(f'{i}. {desc} (우선순위: {priority})')
except Exception as e:
    sys.exit(1)
"
    fi
}

# 기존 데이터셋 파일 확인
check_existing_datasets() {
    local output_dir="$1"
    local found_files=()
    
    echo -e "${BLUE}🔍 기존 데이터셋 파일 확인${NC}"
    
    # 사전훈련 데이터 확인
    if [ -f "$output_dir/korean_pretraining_corpus.json" ]; then
        local size=$(du -h "$output_dir/korean_pretraining_corpus.json" | cut -f1)
        local count=$(python3 -c "import json; data=json.load(open('$output_dir/korean_pretraining_corpus.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "  ✅ 한국어 사전훈련: $size ($count개 문서)"
        found_files+=("korean_pretraining")
    fi
    
    if [ -f "$output_dir/english_pretraining_corpus.json" ]; then
        local size=$(du -h "$output_dir/english_pretraining_corpus.json" | cut -f1)
        local count=$(python3 -c "import json; data=json.load(open('$output_dir/english_pretraining_corpus.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "  ✅ 영어 사전훈련: $size ($count개 문서)"
        found_files+=("english_pretraining")
    fi
    
    if [ -f "$output_dir/mixed_pretraining_corpus.json" ]; then
        local size=$(du -h "$output_dir/mixed_pretraining_corpus.json" | cut -f1)
        local count=$(python3 -c "import json; data=json.load(open('$output_dir/mixed_pretraining_corpus.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "  ✅ 혼합 사전훈련: $size ($count개 문서)"
        found_files+=("mixed_pretraining")
    fi
    
    # 파인튜닝 데이터 확인
    if [ -f "$output_dir/korean_instructions.json" ]; then
        local size=$(du -h "$output_dir/korean_instructions.json" | cut -f1)
        local count=$(python3 -c "import json; data=json.load(open('$output_dir/korean_instructions.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "  ✅ 한국어 파인튜닝: $size ($count개 예시)"
        found_files+=("korean_finetuning")
    fi
    
    if [ -f "$output_dir/english_instructions.json" ]; then
        local size=$(du -h "$output_dir/english_instructions.json" | cut -f1)
        local count=$(python3 -c "import json; data=json.load(open('$output_dir/english_instructions.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "  ✅ 영어 파인튜닝: $size ($count개 예시)"
        found_files+=("english_finetuning")
    fi
    
    if [ ${#found_files[@]} -eq 0 ]; then
        echo "  ❌ 기존 데이터셋 파일이 없습니다."
        return 1
    else
        echo ""
        echo -e "${YELLOW}⚠️  기존 데이터셋이 발견되었습니다 (${#found_files[@]}개 파일)${NC}"
        return 0
    fi
}

# 고유한 파일명 생성
generate_unique_filename() {
    local base_name="$1"
    local extension="$2"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    echo "${base_name}_${timestamp}.${extension}"
}

# 저장 위치 확인 및 설정
setup_output_directory() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local default_root="$(cd "$script_dir/../../../.." && pwd)"
    local default_output="$default_root/datasets"
    
    echo -e "${BLUE}📁 데이터 저장 위치 설정${NC}"
    echo -e "프로젝트 루트: ${YELLOW}$default_root${NC}"
    echo -e "데이터셋 디렉토리: ${YELLOW}$default_output${NC}"
    
    # 기존 파일 확인
    if [ -d "$default_output" ] && check_existing_datasets "$default_output"; then
        echo ""
        echo "기존 데이터를 어떻게 처리하시겠습니까?"
        echo "1) 기존 파일 유지하고 새 파일은 고유한 이름으로 저장"
        echo "2) 기존 파일 덮어쓰기"
        echo "3) 다른 저장 위치 지정"
        echo ""
        
        while true; do
            read -p "선택 (1-3): " existing_choice
            case $existing_choice in
                1)
                    USE_UNIQUE_NAMES=true
                    echo -e "${GREEN}✅ 고유한 파일명으로 저장됩니다.${NC}"
                    break
                    ;;
                2)
                    FORCE_FLAG="--force"
                    echo -e "${YELLOW}⚠️  기존 파일을 덮어씁니다.${NC}"
                    break
                    ;;
                3)
                    echo "새로운 저장 위치를 선택합니다..."
                    break
                    ;;
                *)
                    echo -e "${RED}❌ 1-3 중에서 선택해주세요.${NC}"
                    ;;
            esac
        done
        
        if [ "$existing_choice" != "3" ]; then
            OUTPUT_DIR="$default_output"
            PROJECT_ROOT="$default_root"
            mkdir -p "$OUTPUT_DIR"
            mkdir -p "$PROJECT_ROOT/models"
            echo -e "${GREEN}✅ 저장 위치 확정: $OUTPUT_DIR${NC}"
            return
        fi
    fi
    
    while true; do
        echo ""
        echo "저장 위치를 변경하시겠습니까?"
        echo "1) 기본 위치 사용 ($default_output)"
        echo "2) 다른 위치 지정"
        echo ""
        read -p "선택 (1-2): " choice
        
        case $choice in
            1)
                OUTPUT_DIR="$default_output"
                PROJECT_ROOT="$default_root"
                break
                ;;
            2)
                read -p "새로운 저장 경로를 입력하세요: " custom_path
                if [ -n "$custom_path" ]; then
                    # 상대 경로를 절대 경로로 변환
                    if [[ "$custom_path" != /* ]]; then
                        custom_path="$(pwd)/$custom_path"
                    fi
                    OUTPUT_DIR="$custom_path"
                    PROJECT_ROOT="$(dirname "$custom_path")"
                    break
                else
                    echo -e "${RED}❌ 유효한 경로를 입력해주세요.${NC}"
                fi
                ;;
            *)
                echo -e "${RED}❌ 1 또는 2를 선택해주세요.${NC}"
                ;;
        esac
    done
    
    # 디렉토리 생성
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$PROJECT_ROOT/models"
    
    echo -e "${GREEN}✅ 저장 위치 설정: $OUTPUT_DIR${NC}"
}

# 데이터셋 유형 선택
select_dataset_type() {
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

# 개별 데이터셋 선택 (사전훈련용)
select_pretraining_datasets() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local korean_config="$script_dir/../../configs/training/korean_datasets.json"
    local english_config="$script_dir/../../configs/training/english_datasets.json"
    
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

# 다운로드 옵션 설정
setup_download_options() {
    echo -e "\n${BLUE}⚙️  다운로드 옵션 설정${NC}"
    
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

# 선택사항 요약 표시
show_download_summary() {
    echo -e "\n${BLUE}📋 다운로드 요약${NC}"
    echo "======================================"
    echo -e "저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    if [ "$DATASET_TYPE" = "pretraining" ] || [ "$DATASET_TYPE" = "both" ]; then
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
    fi
    
    if [ "$DATASET_TYPE" = "finetuning" ] || [ "$DATASET_TYPE" = "both" ]; then
        echo "🎯 파인튜닝 데이터셋:"
        if [ "$CREATE_SAMPLES" = true ]; then
            echo "  ✅ 샘플 데이터 생성"
        fi
        if [ "$PROCESS_EXISTING" = true ]; then
            echo "  ✅ 기존 데이터 처리"
        fi
        echo ""
    fi
    
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

check_disk_space() {
    echo -e "${YELLOW}디스크 공간 확인 중...${NC}"
    
    # 현재 디렉토리의 사용 가능한 공간 (GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    available_gb=$((available_space / 1024 / 1024))
    
    # 필요 공간 (GB)
    if [ "$DOWNLOAD_TYPE" = "korean" ]; then
        required_gb=20
    elif [ "$DOWNLOAD_TYPE" = "english" ]; then
        required_gb=30
    elif [ "$DOWNLOAD_TYPE" = "finetuning" ]; then
        required_gb=1
    elif [ "$SMALL_FLAG" = "--small" ]; then
        required_gb=5
    else
        required_gb=50
    fi
    
    echo "사용 가능한 공간: ${available_gb}GB"
    echo "필요한 공간: ${required_gb}GB"
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        echo -e "${RED}❌ 디스크 공간이 부족합니다!${NC}"
        echo "최소 ${required_gb}GB의 여유 공간이 필요합니다."
        exit 1
    else
        echo -e "${GREEN}✅ 디스크 공간 충분${NC}"
    fi
}

check_environment() {
    echo -e "${YELLOW}환경 확인 중...${NC}"
    
    # Python 확인
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3이 설치되지 않았습니다${NC}"
        exit 1
    fi
    
    # 필요한 패키지 확인 (optional)
    python3 -c "import json, pathlib" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 필요한 Python 패키지가 없습니다${NC}"
        echo "다음 명령어로 환경을 설정하세요:"
        echo "conda env create -f conda.yaml"
        echo "conda activate transformers_scratch"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 환경 확인 완료${NC}"
}

# 자동 모드 다운로드
download_datasets_auto() {
    echo -e "${BLUE}자동 모드: 데이터셋 다운로드 시작...${NC}"
    echo "다운로드 타입: $DOWNLOAD_TYPE"
    
    # 프로젝트 루트에서 실행되도록 보장
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    
    # datasets 및 models 디렉토리 생성 (프로젝트 루트에)
    mkdir -p "$PROJECT_ROOT/datasets"
    mkdir -p "$PROJECT_ROOT/models"
    
    case "$DOWNLOAD_TYPE" in
        "korean")
            echo -e "${GREEN}한국어 사전훈련 데이터셋 다운로드 중...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" --korean_only $FORCE_FLAG $SMALL_FLAG
            ;;
        "english")
            echo -e "${GREEN}영어 사전훈련 데이터셋 다운로드 중...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" --english_only $FORCE_FLAG $SMALL_FLAG
            ;;
        "pretraining")
            echo -e "${GREEN}사전훈련용 데이터셋 다운로드 중...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" $FORCE_FLAG $SMALL_FLAG
            ;;
        "finetuning")
            echo -e "${GREEN}파인튜닝용 데이터셋 생성 중...${NC}"
            python3 "$SCRIPT_DIR/download_finetuning.py" --create_samples
            ;;
        "auto")
            echo -e "${GREEN}모든 데이터셋 자동 다운로드 중...${NC}"
            python3 "$SCRIPT_DIR/download_pretraining.py" $FORCE_FLAG $SMALL_FLAG
            if [ $? -eq 0 ]; then
                python3 "$SCRIPT_DIR/download_finetuning.py" --create_samples
            fi
            ;;
    esac
}

# 개별 데이터셋 다운로드 여부 확인
should_download_dataset() {
    local dataset_type="$1"  # korean_pretraining, english_pretraining, etc.
    local output_dir="$2"
    
    case "$dataset_type" in
        "korean_pretraining")
            local file_path="$output_dir/korean_pretraining_corpus.json"
            ;;
        "english_pretraining")
            local file_path="$output_dir/english_pretraining_corpus.json"
            ;;
        "mixed_pretraining")
            local file_path="$output_dir/mixed_pretraining_corpus.json"
            ;;
        "korean_finetuning")
            local file_path="$output_dir/korean_instructions.json"
            ;;
        "english_finetuning")
            local file_path="$output_dir/english_instructions.json"
            ;;
        *)
            return 0  # 알 수 없는 타입은 다운로드
            ;;
    esac
    
    if [ -f "$file_path" ] && [ "$FORCE_FLAG" != "--force" ]; then
        echo -e "${YELLOW}⏭️  $dataset_type 파일이 이미 존재합니다: $file_path${NC}"
        if [ "$USE_UNIQUE_NAMES" = true ]; then
            echo -e "${GREEN}📝 고유한 이름으로 새 파일을 생성합니다.${NC}"
            return 0  # 고유한 이름으로 생성
        else
            echo -e "${BLUE}⏭️  다운로드를 건너뜁니다.${NC}"
            return 1  # 다운로드 건너뛰기
        fi
    fi
    
    return 0  # 다운로드 진행
}

# 대화형 모드 다운로드
download_datasets_interactive() {
    echo -e "\n${BLUE}🚀 데이터셋 다운로드 시작...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # 환경 변수 설정 (고유한 파일명 사용 시)
    local extra_args=""
    if [ "$USE_UNIQUE_NAMES" = true ]; then
        extra_args="--unique_names"
    fi
    
    # 사전훈련 데이터셋 처리
    if [ "$DATASET_TYPE" = "pretraining" ] || [ "$DATASET_TYPE" = "both" ]; then
        echo -e "\n${GREEN}📥 사전훈련 데이터셋 다운로드 중...${NC}"
        
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
    fi
    
    # 파인튜닝 데이터셋 처리
    if [ "$DATASET_TYPE" = "finetuning" ] || [ "$DATASET_TYPE" = "both" ]; then
        echo -e "\n${GREEN}🎯 파인튜닝 데이터셋 처리 중...${NC}"
        
        # 샘플 데이터 생성
        if [ "$CREATE_SAMPLES" = true ]; then
            local korean_needed=true
            local english_needed=true
            
            # 개별 파일 존재 여부 확인
            if ! should_download_dataset "korean_finetuning" "$OUTPUT_DIR"; then
                korean_needed=false
            fi
            if ! should_download_dataset "english_finetuning" "$OUTPUT_DIR"; then
                english_needed=false
            fi
            
            if [ "$korean_needed" = true ] || [ "$english_needed" = true ]; then
                echo -e "${YELLOW}📝 샘플 명령어 데이터 생성...${NC}"
                python3 "$SCRIPT_DIR/download_finetuning.py" --create_samples $extra_args
            fi
        fi
        
        # 기존 데이터 처리
        if [ "$PROCESS_EXISTING" = true ]; then
            echo -e "${YELLOW}🔄 기존 데이터 처리 및 병합...${NC}"
            python3 "$SCRIPT_DIR/download_finetuning.py" --process $extra_args
        fi
    fi
}
    
# 다운로드 결과 표시
show_download_results() {
    local output_dir="${OUTPUT_DIR:-$PROJECT_ROOT/datasets}"
    
    echo -e "\n${GREEN}✅ 데이터셋 다운로드 완료!${NC}"
    
    # 다운로드된 파일 정보 표시
    echo -e "\n${BLUE}📊 다운로드된 데이터:${NC}"
    echo "저장 위치: $output_dir"
    echo ""
    
    local files_found=false
    
    # 사전훈련 데이터 확인
    if [ -f "$output_dir/korean_pretraining_corpus.json" ]; then
        korean_size=$(du -h "$output_dir/korean_pretraining_corpus.json" | cut -f1)
        korean_count=$(python3 -c "import json; data=json.load(open('$output_dir/korean_pretraining_corpus.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "🇰🇷 한국어 사전훈련 데이터: $korean_size ($korean_count개 문서)"
        files_found=true
    fi
    
    if [ -f "$output_dir/english_pretraining_corpus.json" ]; then
        english_size=$(du -h "$output_dir/english_pretraining_corpus.json" | cut -f1)
        english_count=$(python3 -c "import json; data=json.load(open('$output_dir/english_pretraining_corpus.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "🇺🇸 영어 사전훈련 데이터: $english_size ($english_count개 문서)"
        files_found=true
    fi
    
    if [ -f "$output_dir/mixed_pretraining_corpus.json" ]; then
        mixed_size=$(du -h "$output_dir/mixed_pretraining_corpus.json" | cut -f1)
        mixed_count=$(python3 -c "import json; data=json.load(open('$output_dir/mixed_pretraining_corpus.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "🔀 혼합 사전훈련 데이터: $mixed_size ($mixed_count개 문서)"
        files_found=true
    fi
    
    if [ -f "$output_dir/mixed_pretraining_corpus_small.json" ]; then
        mixed_small_size=$(du -h "$output_dir/mixed_pretraining_corpus_small.json" | cut -f1)
        mixed_small_count=$(python3 -c "import json; data=json.load(open('$output_dir/mixed_pretraining_corpus_small.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "🔀 혼합 사전훈련 데이터 (소량): $mixed_small_size ($mixed_small_count개 문서)"
        files_found=true
    fi
    
    # 파인튜닝 데이터 확인
    if [ -f "$output_dir/korean_instructions.json" ]; then
        korean_inst_size=$(du -h "$output_dir/korean_instructions.json" | cut -f1)
        korean_inst_count=$(python3 -c "import json; data=json.load(open('$output_dir/korean_instructions.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "🎯 한국어 파인튜닝 데이터: $korean_inst_size ($korean_inst_count개 예시)"
        files_found=true
    fi
    
    if [ -f "$output_dir/english_instructions.json" ]; then
        english_inst_size=$(du -h "$output_dir/english_instructions.json" | cut -f1)
        english_inst_count=$(python3 -c "import json; data=json.load(open('$output_dir/english_instructions.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "🎯 영어 파인튜닝 데이터: $english_inst_size ($english_inst_count개 예시)"
        files_found=true
    fi
    
    if [ -f "$output_dir/processed_finetuning_data.json" ]; then
        processed_size=$(du -h "$output_dir/processed_finetuning_data.json" | cut -f1)
        processed_count=$(python3 -c "import json; data=json.load(open('$output_dir/processed_finetuning_data.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "🔄 처리된 파인튜닝 데이터: $processed_size ($processed_count개 예시)"
        files_found=true
    fi
    
    if [ "$files_found" = false ]; then
        echo "❌ 다운로드된 파일이 없습니다."
    fi
    
    echo ""
    echo -e "${YELLOW}🚀 다음 단계:${NC}"
    echo "1. 데이터 확인: python3 scripts/dataset/check_datasets.py --data_dir $output_dir"
    echo "2. 토크나이저 훈련: python3 tokenizer.py"
    echo "3. 모델 훈련 시작: python3 training.py --config configs/training/small_model.json"
    echo ""
    echo -e "${BLUE}💡 팁: 생성된 설정을 configs/training/에서 확인하세요!${NC}"
}

# 기본값 설정
DOWNLOAD_TYPE=""
FORCE_FLAG=""
SMALL_FLAG=""
CHECK_ONLY=false
INTERACTIVE_MODE=true
USE_UNIQUE_NAMES=false

# 대화형 모드 변수
DATASET_TYPE=""
DOWNLOAD_KOREAN=false
DOWNLOAD_ENGLISH=false
CREATE_MIXED=false
CREATE_SAMPLES=false
PROCESS_EXISTING=false
OUTPUT_DIR=""
PROJECT_ROOT=""

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
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# 대화형 메인 함수
main_interactive() {
    print_banner
    
    echo -e "${GREEN}🖱️  대화형 모드로 데이터셋 다운로드를 시작합니다!${NC}"
    echo ""
    
    # 환경 확인
    check_environment
    
    # 1단계: 자동 모드 선택 옵션
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
                DOWNLOAD_TYPE="auto"
                INTERACTIVE_MODE=false
                break
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
    
    if [ "$INTERACTIVE_MODE" = false ]; then
        # 자동 모드로 전환
        main_auto
        return
    fi
    
    # 2단계: 저장 위치 설정
    setup_output_directory
    
    # 3단계: 데이터셋 유형 선택
    select_dataset_type
    
    # 4단계: 세부 데이터셋 선택
    if [ "$DATASET_TYPE" = "pretraining" ] || [ "$DATASET_TYPE" = "both" ]; then
        select_pretraining_datasets
    fi
    
    if [ "$DATASET_TYPE" = "finetuning" ] || [ "$DATASET_TYPE" = "both" ]; then
        select_finetuning_datasets
    fi
    
    # 5단계: 다운로드 옵션 설정
    setup_download_options
    
    # 6단계: 선택사항 요약
    show_download_summary
    
    # 7단계: 최종 확인
    echo ""
    echo -e "${YELLOW}위 설정으로 다운로드를 시작하시겠습니까?${NC}"
    read -p "(y/N): " final_choice
    if [[ ! "$final_choice" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    # 8단계: 디스크 공간 확인
    check_disk_space
    
    # 9단계: 다운로드 실행
    download_datasets_interactive
    
    # 10단계: 결과 표시
    show_download_results
}

# 자동 모드 메인 함수
main_auto() {
    print_banner
    
    echo -e "${GREEN}🤖 자동 모드로 데이터셋 다운로드를 시작합니다!${NC}"
    
    if [ "$CHECK_ONLY" = true ]; then
        check_disk_space
        exit 0
    fi
    
    # 기본 설정값 적용
    if [ -z "$OUTPUT_DIR" ]; then
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PROJECT_ROOT="$(cd "$script_dir/../../../.." && pwd)"
        OUTPUT_DIR="$PROJECT_ROOT/datasets"
        mkdir -p "$OUTPUT_DIR"
        mkdir -p "$PROJECT_ROOT/models"
    fi
    
    echo -e "저장 위치: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    check_environment
    check_disk_space
    
    echo -e "${YELLOW}계속하시겠습니까? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    download_datasets_auto
    
    if [ $? -eq 0 ]; then
        show_download_results
    else
        echo -e "${RED}❌ 데이터셋 다운로드 실패${NC}"
        exit 1
    fi
}

# 메인 실행 함수
main() {
    if [ "$INTERACTIVE_MODE" = true ]; then
        main_interactive
    else
        main_auto
    fi
}

# 스크립트가 직접 실행될 때만 main 함수 호출
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 