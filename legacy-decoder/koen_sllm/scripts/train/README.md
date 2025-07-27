# 한국어 sLLM H100 듀얼 GPU 학습 시스템

H100 GPU 2장을 활용한 한국어 소형 언어 모델(sLLM) 분산 학습 시스템입니다.

## 🚀 특징

### **H100 최적화**
- **BF16 Mixed Precision**: H100의 BF16 성능 최적화
- **torch.compile**: PyTorch 2.0+ 컴파일 최적화
- **Flash Attention**: 메모리 효율적인 어텐션 계산
- **Distributed Data Parallel (DDP)**: 2장 GPU 분산 학습

### **유연한 학습 모드**
- **테스트 모드**: 10스텝 빠른 검증 (기본값)
- **실제 학습 모드**: 전체 에포크 학습
- **백그라운드 실행**: 기본값으로 nohup 백그라운드 실행
- **커스터마이징 가능**: 하이퍼파라미터 조정 지원

### **자동 데이터셋 로딩**
- 전처리된 데이터셋 자동 로드 (`../../../../datasets/mixed_pretraining.jsonl`)
- 데이터 없을 시 더미 데이터 자동 생성
- 한국어+영어 혼합 데이터 지원

### **지능형 모델 저장**
- **초기 모델 저장**: 학습 전 초기 파라미터 자동 저장
- **주기적 저장**: 설정 가능한 스텝 간격 (기본 500스텝)
- **상세 메타데이터**: 각 체크포인트에 학습 정보 포함

### **고급 로깅 시스템**
- **다중 로그 파일**: 콘솔, 상세, rank별 로그 분리
- **실시간 모니터링**: tail -f로 학습 진행 상황 추적
- **백그라운드 관리**: PID 파일로 프로세스 추적

## 📁 파일 구조

```
legacy-decoder/koen_sllm/scripts/train/
├── train_h100_dual.py     # H100 듀얼 GPU 학습 스크립트
├── run_h100_dual.sh       # 학습 실행 쉘 스크립트
├── README.md              # 이 파일
├── outputs/               # 학습 결과 출력 (자동 생성)
│   ├── initial_model/     # 초기 모델 (학습 전)
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── model_info.json
│   ├── checkpoint-500/    # 주기적 체크포인트
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── training_info.json
│   └── checkpoint-1000/
└── logs/                  # 상세 로깅 (자동 생성)
    ├── training_console.log      # 메인 콘솔 로그
    ├── training_rank_0.log       # GPU 0 상세 로그
    ├── training_rank_1.log       # GPU 1 상세 로그
    ├── training_detail_rank_0.log # GPU 0 초상세 로그
    ├── training_detail_rank_1.log # GPU 1 초상세 로그
    └── training.pid              # 백그라운드 프로세스 ID
```

## 🔧 설치 요구사항

### **필수 패키지**
```bash
# PyTorch 2.0+ (CUDA 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 추가 패키지
pip install transformers datasets accelerate
```

### **하드웨어 요구사항**
- **GPU**: H100 2장 (최소)
- **VRAM**: 각 GPU당 80GB (권장)
- **RAM**: 64GB+ (권장)
- **CUDA**: 11.8+ 또는 12.0+

## 🚀 빠른 시작

### **1. 테스트 모드 (10스텝)**
```bash
cd legacy-decoder/koen_sllm/scripts/train
chmod +x run_h100_dual.sh
./run_h100_dual.sh --test
```

### **2. 실제 학습 모드**
```bash
./run_h100_dual.sh --train
```

### **3. 커스텀 설정**
```bash
# 더 긴 테스트 (50스텝)
./run_h100_dual.sh --test --max-steps 50

# 배치 크기 조정
./run_h100_dual.sh --train --batch-size 8

# 커스텀 데이터 경로
./run_h100_dual.sh --train --dataset-path /path/to/datasets
```

## 📋 주요 옵션

### **실행 모드**
| 옵션 | 설명 | 기본값 |
|------|------|---------|
| `--test` | 테스트 모드 (빠른 검증) | ✅ |
| `--train` | 실제 학습 모드 | |
| `--max-steps N` | 테스트 모드 최대 스텝 | 10 |

### **데이터 및 출력**
| 옵션 | 설명 | 기본값 |
|------|------|---------|
| `--dataset-path PATH` | 데이터셋 경로 | `../../../../datasets` |
| `--output-dir PATH` | 출력 디렉토리 | `./outputs` |
| `--logs-dir PATH` | 로그 디렉토리 | `./logs` |

### **학습 하이퍼파라미터**
| 옵션 | 설명 | 기본값 |
|------|------|---------|
| `--batch-size N` | 배치 크기 | 4 |
| `--learning-rate LR` | 학습률 | 1e-4 |
| `--max-seq-length N` | 최대 시퀀스 길이 | 2048 |
| `--save-steps N` | 모델 저장 간격 (스텝) | 500 |

### **H100 최적화**
| 옵션 | 설명 | 기본값 |
|------|------|---------|
| `--fp16` | FP16 mixed precision | |
| `--fp32` | FP32 precision | |
| `--no-compile` | torch.compile 비활성화 | |
| `--num-gpus N` | 사용할 GPU 수 | 2 |

### **실행 제어**
| 옵션 | 설명 | 기본값 |
|------|------|---------|
| `--foreground` | 포그라운드에서 실행 | 백그라운드 |
| `--no-save-initial` | 초기 모델 저장 비활성화 | 저장함 |
| `--dry-run` | 명령어만 출력 (실행 안 함) | |
| `--verbose` | 상세 로그 출력 | |

## 💡 사용 예시

### **기본 테스트 (10스텝, 백그라운드)**
```bash
./run_h100_dual.sh
```
**출력 예시:**
```
========================================================
     한국어 sLLM H100 듀얼 GPU 학습 실행기
     Korean sLLM H100 Dual GPU Training Launcher
     🚀 H100 최적화 + 분산 학습 지원
========================================================

🔍 H100 학습 환경 확인 중...
✅ PyTorch 및 CUDA 확인 완료
✅ GPU 확인 완료: 2개 사용 가능
  GPU 0: NVIDIA H100 PCIe
  GPU 1: NVIDIA H100 PCIe
✅ 데이터셋 확인 완료: 1개 파일
✅ 모든 요구사항 충족

📋 학습 정보
=================================
🎯 모드: 테스트 모드 (10스텝)
🔧 하이퍼파라미터:
   • 배치 크기: 4
   • 학습률: 1e-4
   • 최대 시퀀스 길이: 2048
   • Mixed Precision: bf16
   • 모델 컴파일: true
📁 경로:
   • 데이터셋: ../../../../datasets
   • 출력: ./outputs
   • 로그: ./logs
💾 저장 설정:
   • 저장 간격: 500 스텝
   • 초기 모델 저장: true
   • 백그라운드 실행: true
🚀 H100 설정:
   • GPU 수: 2
   • 분산 학습: DDP
   • 최적화: BF16 + torch.compile

🎯 학습 시작!
🔄 백그라운드 모드로 실행됩니다
📋 로그 파일: ./logs/training_console.log
📋 실시간 로그 확인: tail -f ./logs/training_console.log

✅ 백그라운드 학습 시작됨!
🔢 프로세스 ID: 12345
📄 PID 파일: ./logs/training.pid

🔍 학습 상태 확인 명령어:
  tail -f ./logs/training_console.log     # 실시간 로그
  ps -p 12345                             # 프로세스 상태
  kill 12345                              # 학습 중단

📋 초기 로그 출력 (5초간):
----------------------------------------
🚀 초기 모델 저장 중...
✅ 초기 모델 저장 완료: ./outputs/initial_model
🚀 학습 시작!
테스트 모드: True
최대 스텝: 10
----------------------------------------
백그라운드에서 계속 실행 중...
```

### **성능 최적화 테스트**
```bash
# 더 큰 배치 크기로 처리량 최적화
./run_h100_dual.sh --test --batch-size 8 --max-steps 20

# FP16으로 메모리 절약
./run_h100_dual.sh --test --fp16 --batch-size 12
```

### **실제 학습 시작**
```bash
# 기본 설정으로 학습
./run_h100_dual.sh --train

# 최적화된 설정으로 학습  
./run_h100_dual.sh --train --batch-size 6 --learning-rate 5e-5
```

### **명령어 확인만**
```bash
# 실제 실행 없이 명령어만 확인
./run_h100_dual.sh --dry-run --train --batch-size 8
```

## 📊 성능 모니터링

### **학습 로그 확인**
```bash
# 실시간 로그 확인
tail -f outputs/training.log

# GPU 사용률 모니터링
nvidia-smi -l 1
```

### **체크포인트 관리**
```bash
# 저장된 체크포인트 확인
ls -la outputs/checkpoint-*

# 특정 체크포인트 로드
python3 ../../inference.py \
  --model-path outputs/checkpoint-10/pytorch_model.bin
```

## 🔧 트러블슈팅

### **CUDA 메모리 부족**
```bash
# 배치 크기 감소
./run_h100_dual.sh --test --batch-size 2

# FP16 사용
./run_h100_dual.sh --test --fp16
```

### **분산 학습 오류**
```bash
# 단일 GPU로 테스트
./run_h100_dual.sh --test --num-gpus 1

# 컴파일 비활성화
./run_h100_dual.sh --test --no-compile
```

### **데이터셋 없음**
```bash
# 데이터 전처리 먼저 실행 (프로젝트 루트에서)
cd ../../../../
./legacy-decoder/koen_sllm/scripts/dataset/run_preprocessing.sh --recipe1

# 다시 학습 시도
cd legacy-decoder/koen_sllm/scripts/train
./run_h100_dual.sh --test
```

## 🎯 다음 단계

### **모델 성능 검증**
1. **테스트 모드 완료** → 실제 학습 시작
2. **체크포인트 저장** → 추론 성능 테스트  
3. **벤치마크 평가** → 모델 품질 검증

### **고급 설정**
- **Gradient Accumulation**: 더 큰 effective batch size
- **Learning Rate Scheduling**: Cosine annealing 등
- **Mixed Precision**: BF16/FP16 성능 비교

### **배포 준비**
- **모델 양자화**: 추론 최적화
- **ONNX 변환**: 다양한 플랫폼 지원
- **서빙 최적화**: TensorRT, vLLM 등

---

## 🆘 도움말

```bash
# 전체 옵션 확인
./run_h100_dual.sh --help

# 환경 확인만
./run_h100_dual.sh --dry-run
```

**문제가 발생하면:**
1. GPU 메모리 상태 확인: `nvidia-smi`
2. CUDA 설치 확인: `nvcc --version`
3. PyTorch CUDA 확인: `python3 -c "import torch; print(torch.cuda.is_available())"` 