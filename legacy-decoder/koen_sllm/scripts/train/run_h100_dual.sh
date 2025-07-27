#!/bin/bash

# 한국어 sLLM H100 듀얼 GPU 학습 실행 스크립트
# Korean sLLM Training Script for Dual H100 GPUs

set -e  # 에러 발생 시 스크립트 종료

# =============================================================================
# 스크립트 설정
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 기본 설정
MODE="test"  # test 또는 train
DATASET_PATH="../../../../datasets"
OUTPUT_DIR="./outputs"
LOGS_DIR="./logs"
BATCH_SIZE=4
LEARNING_RATE=1e-4
MAX_SEQ_LENGTH=2048
MAX_STEPS=10
NUM_GPUS=2
SAVE_STEPS=500

# 실행 설정
BACKGROUND=true  # 기본 백그라운드 실행
SAVE_INITIAL=true  # 초기 모델 저장

# H100 최적화 설정
MIXED_PRECISION="bf16"  # H100은 bf16 최적화
COMPILE_MODEL=true
FLASH_ATTENTION=true

print_banner() {
    echo -e "${PURPLE}"
    echo "========================================================"
    echo "     한국어 sLLM H100 듀얼 GPU 학습 실행기"
    echo "     Korean sLLM H100 Dual GPU Training Launcher"
    echo "     🚀 H100 최적화 + 분산 학습 지원"
    echo "========================================================"
    echo -e "${NC}"
}

print_help() {
    echo "한국어 sLLM H100 듀얼 GPU 학습 실행기"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "🎯 실행 모드:"
    echo "  --test               테스트 모드 (10스텝, 기본값)"
    echo "  --train              실제 학습 모드"
    echo "  --max-steps N        테스트 모드 최대 스텝 (기본값: 10)"
    echo ""
    echo "📁 데이터 및 출력:"
    echo "  --dataset-path PATH  데이터셋 경로 (기본값: ../../../../datasets)"
    echo "  --output-dir PATH    출력 디렉토리 (기본값: ./outputs)"
    echo "  --logs-dir PATH      로그 디렉토리 (기본값: ./logs)"
    echo ""
    echo "🔧 학습 하이퍼파라미터:"
    echo "  --batch-size N       배치 크기 (기본값: 4)"
    echo "  --learning-rate LR   학습률 (기본값: 1e-4)"
    echo "  --max-seq-length N   최대 시퀀스 길이 (기본값: 2048)"
    echo "  --save-steps N       모델 저장 간격 (기본값: 500)"
    echo ""
    echo "🚀 H100 최적화:"
    echo "  --fp16               FP16 mixed precision 사용"
    echo "  --fp32               FP32 precision 사용"
    echo "  --no-compile         torch.compile 비활성화"
    echo "  --num-gpus N         사용할 GPU 수 (기본값: 2)"
    echo ""
    echo "🔍 분석 및 디버깅:"
    echo "  --dry-run            실제 실행 없이 명령어만 출력"
    echo "  --verbose            상세한 로그 출력"
    echo "  --foreground         포그라운드에서 실행 (기본값: 백그라운드)"
    echo "  --no-save-initial    초기 모델 저장 비활성화"
    echo ""
    echo "예시:"
    echo "  $0 --test                           # 테스트 모드 (기본)"
    echo "  $0 --test --max-steps 20            # 20스텝 테스트"
    echo "  $0 --train --batch-size 8           # 실제 학습"
    echo "  $0 --train --dataset-path ./data    # 커스텀 데이터 경로"
    echo "  $0 --dry-run                        # 명령어만 확인"
    echo ""
    echo "🎯 H100 최적화 기능:"
    echo "  • BF16 Mixed Precision (기본값)"
    echo "  • torch.compile 최적화"
    echo "  • Flash Attention 지원"
    echo "  • Distributed Data Parallel (DDP)"
    echo "  • Gradient Accumulation"
    echo ""
}

check_requirements() {
    echo -e "${YELLOW}🔍 H100 학습 환경 확인 중...${NC}"
    
    local requirements_met=true
    
    # Dry-run 모드일 때는 요구사항 체크를 더 관대하게
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}📝 Dry-run 모드: 기본 요구사항만 체크${NC}"
    fi
    
    # Python 및 PyTorch 확인
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3가 설치되지 않았습니다${NC}"
        requirements_met=false
    fi
    
    # PyTorch CUDA 확인
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}⚠️  Dry-run: PyTorch/CUDA 체크 스킵${NC}"
    else
        if ! python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
            echo -e "${RED}❌ PyTorch 또는 CUDA가 제대로 설치되지 않았습니다${NC}"
            requirements_met=false
        else
            echo -e "${GREEN}✅ PyTorch 및 CUDA 확인 완료${NC}"
        fi
    fi
    
    # GPU 개수 확인
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}⚠️  Dry-run: GPU 체크 스킵 (요구: $NUM_GPUS개)${NC}"
    else
        local available_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        if [ "$available_gpus" -lt "$NUM_GPUS" ]; then
            echo -e "${RED}❌ 요구 GPU 수: $NUM_GPUS, 사용 가능: $available_gpus${NC}"
            requirements_met=false
        else
            echo -e "${GREEN}✅ GPU 확인 완료: $available_gpus개 사용 가능${NC}"
            
            # GPU 정보 출력
            python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    print(f'  GPU {i}: {name}')
" 2>/dev/null || true
        fi
    fi
    
    # 데이터셋 확인
    if [ ! -d "$DATASET_PATH" ]; then
        echo -e "${YELLOW}⚠️  데이터셋 디렉토리가 없습니다: $DATASET_PATH${NC}"
        echo "데이터 전처리 스크립트를 먼저 실행하세요."
    else
        local data_files=$(find "$DATASET_PATH" -name "*.jsonl" | wc -l)
        if [ $data_files -gt 0 ]; then
            echo -e "${GREEN}✅ 데이터셋 확인 완료: $data_files개 파일${NC}"
        else
            echo -e "${YELLOW}⚠️  JSONL 데이터 파일이 없습니다${NC}"
        fi
    fi
    
    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_DIR"
    if [ ! -w "$OUTPUT_DIR" ]; then
        echo -e "${RED}❌ 출력 디렉토리 쓰기 권한이 없습니다: $OUTPUT_DIR${NC}"
        requirements_met=false
    fi
    
    if [ "$requirements_met" = true ]; then
        echo -e "${GREEN}✅ 모든 요구사항 충족${NC}"
        return 0
    else
        echo -e "${RED}❌ 요구사항 미충족${NC}"
        return 1
    fi
}

build_training_command() {
    local cmd="torchrun"
    
    # 분산 학습 설정
    cmd="$cmd --nproc_per_node=$NUM_GPUS"
    cmd="$cmd --nnodes=1"
    cmd="$cmd --node_rank=0"
    cmd="$cmd --master_addr=localhost"
    cmd="$cmd --master_port=29500"
    
    # Python 스크립트 및 인수
    cmd="$cmd $SCRIPT_DIR/train_h100_dual.py"
    
    # 모드 설정
    if [ "$MODE" = "test" ]; then
        cmd="$cmd --test-mode"
        cmd="$cmd --max-steps $MAX_STEPS"
    fi
    
    # 데이터 및 출력
    cmd="$cmd --dataset-path $DATASET_PATH"
    cmd="$cmd --output-dir $OUTPUT_DIR"
    cmd="$cmd --logs-dir $LOGS_DIR"
    
    # 학습 하이퍼파라미터
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --max-seq-length $MAX_SEQ_LENGTH"
    cmd="$cmd --save-steps $SAVE_STEPS"
    
    # H100 최적화
    cmd="$cmd --mixed-precision $MIXED_PRECISION"
    
    if [ "$COMPILE_MODEL" = false ]; then
        cmd="$cmd --no-compile"
    fi
    
    # 초기 모델 저장
    if [ "$SAVE_INITIAL" = false ]; then
        cmd="$cmd --no-save-initial"
    fi
    
    echo "$cmd"
}

print_training_info() {
    echo -e "\n${BLUE}📋 학습 정보${NC}"
    echo "================================="
    
    if [ "$MODE" = "test" ]; then
        echo -e "🎯 모드: ${YELLOW}테스트 모드${NC} (${MAX_STEPS}스텝)"
    else
        echo -e "🎯 모드: ${GREEN}실제 학습 모드${NC}"
    fi
    
    echo -e "🔧 하이퍼파라미터:"
    echo "   • 배치 크기: $BATCH_SIZE"
    echo "   • 학습률: $LEARNING_RATE"
    echo "   • 최대 시퀀스 길이: $MAX_SEQ_LENGTH"
    echo "   • Mixed Precision: $MIXED_PRECISION"
    echo "   • 모델 컴파일: $COMPILE_MODEL"
    
    echo -e "📁 경로:"
    echo "   • 데이터셋: $DATASET_PATH"
    echo "   • 출력: $OUTPUT_DIR"
    echo "   • 로그: $LOGS_DIR"
    
    echo -e "💾 저장 설정:"
    echo "   • 저장 간격: $SAVE_STEPS 스텝"
    echo "   • 초기 모델 저장: $SAVE_INITIAL"
    echo "   • 백그라운드 실행: $BACKGROUND"
    
    echo -e "🚀 H100 설정:"
    echo "   • GPU 수: $NUM_GPUS"
    echo "   • 분산 학습: DDP"
    echo "   • 최적화: BF16 + torch.compile"
    echo ""
}

execute_training() {
    local cmd=$(build_training_command)
    
    print_training_info
    
    echo -e "${GREEN}🚀 학습 실행 명령어:${NC}"
    echo "$cmd"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}📝 Dry run 모드: 실제 실행하지 않습니다${NC}"
        return 0
    fi
    
    # 로그 디렉토리 생성
    mkdir -p "$LOGS_DIR"
    
    echo -e "${GREEN}🎯 학습 시작!${NC}"
    if [ "$BACKGROUND" = true ]; then
        echo -e "${BLUE}🔄 백그라운드 모드로 실행됩니다${NC}"
        echo -e "${YELLOW}📋 로그 파일: $LOGS_DIR/training_console.log${NC}"
        echo -e "${YELLOW}📋 실시간 로그 확인: tail -f $LOGS_DIR/training_console.log${NC}"
    else
        echo "Ctrl+C로 중단할 수 있습니다."
    fi
    echo ""
    
    # 환경 변수 설정
    export CUDA_VISIBLE_DEVICES="0,1"  # H100 2장
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"  # 메모리 최적화
    export TORCH_CUDNN_V8_API_ENABLED=1  # cuDNN 최적화
    
    # 학습 실행
    if [ "$BACKGROUND" = true ]; then
        # 백그라운드 실행
        nohup bash -c "eval '$cmd'" > "$LOGS_DIR/training_console.log" 2>&1 &
        local training_pid=$!
        echo "$training_pid" > "$LOGS_DIR/training.pid"
        
        echo -e "${GREEN}✅ 백그라운드 학습 시작됨!${NC}"
        echo -e "${BLUE}🔢 프로세스 ID: $training_pid${NC}"
        echo -e "${BLUE}📄 PID 파일: $LOGS_DIR/training.pid${NC}"
        echo ""
        echo -e "${YELLOW}🔍 학습 상태 확인 명령어:${NC}"
        echo "  tail -f $LOGS_DIR/training_console.log     # 실시간 로그"
        echo "  ps -p $training_pid                        # 프로세스 상태"
        echo "  kill $training_pid                         # 학습 중단"
        echo ""
        
        # 초기 로그 출력 (5초간)
        echo -e "${BLUE}📋 초기 로그 출력 (5초간):${NC}"
        echo "----------------------------------------"
        timeout 5s tail -f "$LOGS_DIR/training_console.log" 2>/dev/null || true
        echo "----------------------------------------"
        echo -e "${GREEN}백그라운드에서 계속 실행 중...${NC}"
        
        return 0
    else
        # 포그라운드 실행
        eval "$cmd"
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}🎉 학습 완료!${NC}"
        
        # 결과 요약
        if [ -d "$OUTPUT_DIR" ]; then
            echo -e "\n${BLUE}📊 학습 결과:${NC}"
            echo "출력 디렉토리: $OUTPUT_DIR"
            
            # 체크포인트 확인
            local checkpoints=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | wc -l)
            if [ $checkpoints -gt 0 ]; then
                echo "저장된 체크포인트: $checkpoints개"
                find "$OUTPUT_DIR" -name "checkpoint-*" -type d | head -3 | while read checkpoint; do
                    echo "  • $(basename "$checkpoint")"
                done
            fi
            
            # 로그 파일 확인
            local log_files=$(find "$OUTPUT_DIR" -name "*.log" | wc -l)
            if [ $log_files -gt 0 ]; then
                echo "로그 파일: $log_files개"
            fi
        fi
        
        echo -e "\n${YELLOW}🚀 다음 단계:${NC}"
        echo "1. 체크포인트 확인: ls -la $OUTPUT_DIR/checkpoint-*"
        echo "2. 로그 확인: tail -f $OUTPUT_DIR/*.log"
        if [ "$MODE" = "test" ]; then
            echo "3. 실제 학습: $0 --train"
        else
            echo "3. 추론 테스트: python3 inference.py --model-path $OUTPUT_DIR/checkpoint-XXX"
        fi
        
    else
        echo -e "\n${RED}❌ 학습 실패 (종료 코드: $exit_code)${NC}"
        echo "로그를 확인하여 오류를 분석하세요."
        return $exit_code
    fi
}

# =============================================================================
# 명령행 인수 파싱
# =============================================================================

DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            shift
            ;;
        --train)
            MODE="train"
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --logs-dir)
            LOGS_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --fp16)
            MIXED_PRECISION="fp16"
            shift
            ;;
        --fp32)
            MIXED_PRECISION="fp32"
            shift
            ;;
        --no-compile)
            COMPILE_MODEL=false
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --foreground)
            BACKGROUND=false
            shift
            ;;
        --no-save-initial)
            SAVE_INITIAL=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
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

# =============================================================================
# 메인 실행
# =============================================================================

print_banner

# 요구사항 확인
if ! check_requirements; then
    echo -e "${RED}❌ 환경 요구사항을 만족하지 않습니다${NC}"
    exit 1
fi

# 학습 실행
execute_training 