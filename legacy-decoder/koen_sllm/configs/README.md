# Configuration Files Directory

이 디렉토리는 트랜스포머 모델 프로젝트의 모든 설정 파일들을 용도별로 체계적으로 관리합니다.

## 📁 디렉토리 구조

```
configs/
├── training/           # 모델 훈련 설정
│   ├── small_model.json      # 소형 모델 훈련 설정 (124M)
│   ├── base_model.json       # 기본 모델 훈련 설정 (355M)
│   └── large_model.json      # 대형 모델 훈련 설정 (1.3B)
├── models/             # 모델 아키텍처 설정
│   ├── gpt_small.json        # GPT-Small 아키텍처
│   ├── gpt_base.json         # GPT-Base 아키텍처
│   └── gpt_large.json        # GPT-Large 아키텍처
├── dataset/           # 데이터셋 설정
│   ├── pretraining.json      # 사전훈련용 데이터 설정
│   └── finetuning.json       # 파인튜닝용 데이터 설정
└── README.md           # 이 파일
```

## 🎯 용도별 분류

### 1. **Training Configs** (`training/`)
모델 훈련에 필요한 모든 하이퍼파라미터를 포함:
- 학습률, 배치 크기, 스케줄링
- 모델 아키텍처 + 훈련 설정 통합
- GPU/분산 학습 설정

### 2. **Model Architecture Configs** (`models/`)
순수 모델 아키텍처 정의만 포함:
- 모델 구조 (layers, heads, dimensions)
- 아키텍처 하이퍼파라미터
- 훈련과 독립적인 모델 정의

### 3. **Dataset Configs** (`dataset/`)
데이터 처리 및 로딩 설정:
- 데이터셋 경로 및 형식
- 전처리 파이프라인
- 데이터 로더 설정

## 🚀 사용 방법

### 훈련 시작
```bash
# 소형 모델로 빠른 테스트
python train.py --config configs/training/small_model.json

# 기본 모델로 실제 훈련
python train.py --config configs/training/base_model.json

# 대형 모델로 production 훈련
python train.py --config configs/training/large_model.json
```

### 모델 생성
```python
from transformers import AutoConfig, AutoModel

# 모델 아키텍처만 로드
config = AutoConfig.from_json_file("configs/models/gpt_base.json")
model = AutoModel.from_config(config)
```

### 데이터셋 설정
```python
import json

# 사전훈련용 데이터 설정 로드
with open("configs/dataset/pretraining.json") as f:
    data_config = json.load(f)
```

## ⚙️ 설정 파일 선택 가이드

### 모델 크기별 권장사항

| 모델 크기 | 파라미터 수 | Training Config | Model Config | 용도 |
|----------|------------|----------------|---------------|------|
| **Small** | 124M | `small_model.json` | `gpt_small.json` | 빠른 실험, 테스트 |
| **Base** | 355M | `base_model.json` | `gpt_base.json` | 일반적인 훈련 |
| **Large** | 1.3B | `large_model.json` | `gpt_large.json` | Production 모델 |

### 하드웨어별 권장사항

| GPU 메모리 | 권장 설정 | 배치 크기 | 그래디언트 누적 |
|-----------|----------|----------|----------------|
| 8GB | Small | 4-8 | 4-8 |
| 16GB | Base | 2-4 | 8-16 |
| 24GB+ | Large | 1-2 | 16-32 |

## 🔧 설정 파일 수정

### 새로운 모델 크기 추가
1. `models/` 에 새 아키텍처 파일 생성
2. `training/` 에 해당 훈련 설정 생성
3. 파라미터 수 계산하여 `estimated_parameters` 업데이트

### 하이퍼파라미터 튜닝
- **학습률**: `0.0001` → `0.00005` (대형 모델일수록 작게)
- **배치 크기**: 메모리에 맞게 조정
- **시퀀스 길이**: 태스크에 따라 조정

### 분산 학습 설정
- `world_size`: 사용할 GPU 수
- `grad_accumulation_steps`: 효과적 배치 크기 조정
- `deepspeed_config`: DeepSpeed 사용 시 설정 파일 지정

## 📝 설정 검증

### 필수 필드 확인
모든 training config에는 다음이 포함되어야 함:
- 모델 아키텍처 (vocab_size, d_model, n_heads, n_layers)
- 훈련 설정 (learning_rate, batch_size, max_steps)
- 데이터 경로 (train_data_path, val_data_path, tokenizer_path)

### 설정 유효성 검사
```bash
# 설정 파일 문법 검사
python -m json.tool configs/training/base_model.json

# 모델 생성 테스트
python scripts/validate_config.py --config configs/training/base_model.json
```

## 🔗 관련 문서

- [Training Guide](../docs/training.md)
- [Model Architecture](../docs/models.md)
- [Dataset Preparation](../common/scripts/dataset/README.md)
- [Distributed Training](../docs/distributed.md) 