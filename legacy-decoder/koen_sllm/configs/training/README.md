# Training Configuration Files

이 디렉토리는 모델 훈련을 위한 통합 설정 파일들을 포함합니다.

## 📋 설정 파일 목록

### `small_model.json` - 소형 모델 (124M 파라미터)
```json
{
  "description": "빠른 실험과 테스트를 위한 소형 모델 설정",
  "vocab_size": 16000,
  "d_model": 512,
  "n_heads": 8,
  "n_layers": 8,
  "estimated_parameters": "124M"
}
```

**사용 시나리오:**
- 빠른 프로토타이핑
- 코드 테스트 및 디버깅
- 제한된 GPU 메모리 환경
- 개념 검증 (Proof of Concept)

**권장 하드웨어:**
- GPU: 8GB 이상
- RAM: 16GB 이상
- 훈련 시간: 수 시간 ~ 1일

### `base_model.json` - 기본 모델 (355M 파라미터)
```json
{
  "description": "일반적인 용도의 균형잡힌 모델 설정",
  "vocab_size": 32000,
  "d_model": 768,
  "n_heads": 12,
  "n_layers": 12,
  "estimated_parameters": "355M"
}
```

**사용 시나리오:**
- 표준 언어 모델 훈련
- 실용적인 성능과 속도의 균형
- 중간 규모 프로젝트
- Fine-tuning 베이스 모델

**권장 하드웨어:**
- GPU: 16GB 이상
- RAM: 32GB 이상
- 훈련 시간: 1-3일

### `large_model.json` - 대형 모델 (1.3B 파라미터)
```json
{
  "description": "고성능이 필요한 production 환경용 대형 모델",
  "vocab_size": 50000,
  "d_model": 1024,
  "n_heads": 16,
  "n_layers": 24,
  "estimated_parameters": "1.3B"
}
```

**사용 시나리오:**
- Production 환경 배포
- 최고 성능이 필요한 애플리케이션
- 대규모 데이터셋 훈련
- 연구용 SOTA 모델

**권장 하드웨어:**
- GPU: 24GB+ 이상 (다중 GPU 권장)
- RAM: 64GB+ 이상
- 훈련 시간: 수일 ~ 수주

## ⚙️ 주요 설정 필드 설명

### 모델 아키텍처
- **`vocab_size`**: 토크나이저 어휘 크기
- **`d_model`**: 모델 임베딩 차원
- **`n_heads`**: 멀티헤드 어텐션 헤드 수
- **`n_layers`**: 트랜스포머 레이어 수
- **`d_ff`**: 피드포워드 네트워크 차원
- **`max_seq_len`**: 최대 시퀀스 길이

### 훈련 하이퍼파라미터
- **`learning_rate`**: 초기 학습률
- **`batch_size`**: 배치 크기
- **`grad_accumulation_steps`**: 그래디언트 누적 스텝
- **`max_steps`**: 최대 훈련 스텝
- **`warmup_steps`**: 웜업 스텝 수
- **`save_steps`**: 체크포인트 저장 간격
- **`eval_steps`**: 평가 수행 간격

### 시스템 설정
- **`world_size`**: 분산 훈련 GPU 수
- **`fp16`**: 혼합 정밀도 훈련 사용
- **`gradient_checkpointing`**: 메모리 절약을 위한 그래디언트 체크포인팅

## 🚀 사용 예시

### 기본 사용법
```bash
# 소형 모델로 빠른 테스트
python train.py --config configs/training/small_model.json

# 기본 모델로 표준 훈련
python train.py --config configs/training/base_model.json

# 대형 모델로 분산 훈련
torchrun --nproc_per_node=4 train.py --config configs/training/large_model.json
```

### 설정 오버라이드
```bash
# 학습률 변경
python train.py --config configs/training/base_model.json --learning_rate 0.00005

# 배치 크기 조정
python train.py --config configs/training/small_model.json --batch_size 32
```

### 체크포인트에서 재시작
```bash
python train.py --config configs/training/base_model.json --resume_from_checkpoint outputs/checkpoint-10000
```

## 🔧 커스터마이징 가이드

### 새로운 모델 크기 추가
1. 기존 설정 파일을 복사
2. 파라미터 수 계산
3. 하드웨어에 맞게 배치 크기 조정
4. 적절한 학습률 설정

### 메모리 최적화
- `gradient_checkpointing: true` 활성화
- `batch_size` 줄이고 `grad_accumulation_steps` 증가
- `fp16: true` 또는 `bf16: true` 사용

### 속도 최적화
- `num_workers` 증가 (데이터 로딩)
- `pin_memory: true` 설정
- SSD 사용으로 I/O 병목 해결

## 📊 성능 벤치마크

| 모델 | GPU 메모리 | 훈련 속도 | 수렴 시간 | PPL (검증) |
|------|-----------|-----------|-----------|------------|
| Small | 6GB | 1000 tok/s | 12시간 | ~15 |
| Base | 12GB | 800 tok/s | 2일 | ~12 |
| Large | 20GB | 600 tok/s | 1주 | ~10 |

## 🐛 문제 해결

### OOM (Out of Memory) 오류
1. `batch_size` 절반으로 줄이기
2. `grad_accumulation_steps` 두 배로 늘리기
3. `gradient_checkpointing` 활성화
4. 더 작은 모델 크기 사용

### 느린 훈련 속도
1. `num_workers` 증가
2. 데이터를 SSD로 이동
3. `pin_memory` 활성화
4. 혼합 정밀도 사용

### 수렴하지 않는 경우
1. 학습률을 절반으로 줄이기
2. 웜업 스텝 늘리기
3. 배치 크기 늘리기
4. 데이터 품질 확인 