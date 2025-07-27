# Korean Small Language Model (sLLM) from Scratch

한국어 소형 언어모델을 PyTorch로 스크래치부터 구현한 프로젝트입니다. H100 두 장에서 멀티 GPU 학습이 가능하도록 설계되었습니다.

## 🚀 주요 특징

- **순수 PyTorch 구현**: 추상화를 배제하고 torch 레벨에서 직접 구현
- **한국어-영어 지원**: BPE 토크나이저로 두 언어 동시 지원
- **멀티 GPU 학습**: DistributedDataParallel을 이용한 H100 두 장 학습
- **완전한 파이프라인**: 데이터 수집부터 추론까지 전 과정 구현
- **실시간 모니터링**: Weights & Biases 통합

## 📁 프로젝트 구조

```
korean_sllm/
├── korean_sllm/           # 메인 패키지
│   ├── __init__.py
│   ├── config.py          # 모델 설정
│   ├── tokenizer.py       # 한국어-영어 BPE 토크나이저
│   ├── model.py           # GPT 스타일 트랜스포머 모델
│   ├── dataset.py         # 데이터셋 다운로드 및 전처리
│   ├── training.py        # 멀티 GPU 사전학습
│   ├── validation.py      # 모델 검증
│   ├── inference.py       # 텍스트 생성 및 추론
│   └── utils.py           # 유틸리티 함수들
├── common/                # 공통 도구들
│   └── scripts/           # 실행 스크립트
│       ├── train.py           # 학습 실행
│       ├── train_multi_gpu.sh # 멀티 GPU 학습 스크립트
│       ├── inference.py       # 추론 실행
│       ├── validate.py        # 검증 실행
│       ├── download_datasets.sh # 데이터셋 다운로드
│       ├── download_all.py    # 전체 데이터 다운로드
│       ├── download_korean.py # 한국어 데이터 다운로드
│       ├── download_english.py # 영어 데이터 다운로드
│       └── check_datasets.py  # 데이터셋 확인
├── configs/               # 설정 파일들
│   ├── default_config.json
│   └── small_config.json
├── conda.yaml             # 환경 설정
└── README.md
```

## 🛠️ 설치 및 환경 설정

### Conda 환경 생성

```bash
# Conda 환경 생성
conda env create -f conda.yaml
conda activate transformers_scratch
```

### 필요한 패키지

주요 의존성:
- PyTorch 2.7.1+ (CUDA 지원)
- Transformers 4.54.0
- Datasets (HuggingFace)
- Weights & Biases
- regex (고급 정규표현식)

## 🚀 빠른 시작

### 0. 데이터셋 다운로드 (선택적)

```bash
# 편리한 shell 스크립트로 데이터셋 다운로드
./common/scripts/download_datasets.sh --small  # 테스트용 소량
./common/scripts/download_datasets.sh --all    # 전체 데이터

# 개별 언어별 다운로드
./common/scripts/download_datasets.sh --korean   # 한국어만
./common/scripts/download_datasets.sh --english  # 영어만

# 데이터셋 확인
python3 common/scripts/check_datasets.py --show_samples
```

### 1. 토크나이저 학습

```bash
# 토크나이저 학습용 데이터 다운로드 및 학습 (자동)
python common/scripts/train.py --config configs/default_config.json
```

### 2. 멀티 GPU 사전학습

```bash
# H100 두 장을 사용한 분산 학습
chmod +x common/scripts/train_multi_gpu.sh
./common/scripts/train_multi_gpu.sh
```

또는 직접 실행:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr="localhost" \
    --master_port="12345" \
    common/scripts/train.py \
    --config configs/default_config.json \
    --local_rank 0
```

### 3. 모델 검증

```bash
python common/scripts/validate.py \
    --model_path outputs/best_model.pt \
    --tokenizer_path tokenizer \
    --config_path configs/default_config.json
```

### 4. 텍스트 생성

```bash
# 단일 생성
python common/scripts/inference.py \
    --model_path outputs/best_model.pt \
    --tokenizer_path tokenizer \
    --prompt "한국어는" \
    --max_length 100 \
    --temperature 0.8

# 대화형 모드
python common/scripts/inference.py \
    --model_path outputs/best_model.pt \
    --tokenizer_path tokenizer \
    --mode chat

# 벤치마크
python common/scripts/inference.py \
    --model_path outputs/best_model.pt \
    --tokenizer_path tokenizer \
    --mode benchmark
```

## 🔧 설정 옵션

### 모델 아키텍처 설정

```json
{
  "vocab_size": 32000,        // 어휘 크기
  "d_model": 768,             // 임베딩 차원
  "n_heads": 12,              // 어텐션 헤드 수
  "n_layers": 12,             // 트랜스포머 레이어 수
  "d_ff": 3072,               // 피드포워드 네트워크 차원
  "max_seq_len": 2048,        // 최대 시퀀스 길이
  "dropout": 0.1              // 드롭아웃 비율
}
```

### 학습 설정

```json
{
  "learning_rate": 1e-4,      // 학습률
  "batch_size": 8,            // 배치 크기 (GPU당)
  "grad_accumulation_steps": 8, // 그래디언트 누적 스텝
  "max_steps": 100000,        // 최대 학습 스텝
  "warmup_steps": 2000,       // 워밍업 스텝
  "fp16": true,               // Mixed Precision 사용
  "gradient_checkpointing": true // 그래디언트 체크포인팅
}
```

## 📊 모델 성능

### 모델 크기별 사양

| 설정 | 파라미터 수 | 메모리 사용량 | 학습 시간 (H100 x2) |
|------|-------------|---------------|---------------------|
| Small | ~85M | ~8GB | ~24시간 |
| Default | ~350M | ~24GB | ~72시간 |

### 벤치마크 결과

- **생성 속도**: ~50 tokens/sec (H100)
- **Perplexity**: 한국어 ~25, 영어 ~20 (베이스라인 대비)
- **메모리 효율성**: Gradient checkpointing으로 메모리 사용량 50% 절약

## 📊 데이터셋

자동으로 다운로드되는 데이터셋:

### 한국어 데이터
- 한국어 위키피디아
- KLUE 뉴스 데이터
- AI Hub 일상대화 데이터
- 한국어 CommonCrawl

### 영어 데이터
- OpenWebText
- WikiText-103
- Gutenberg Books
- CC-News 영어

총 데이터 크기: ~50GB (압축 후 ~10GB)

## 🔍 주요 구현 특징

### 1. 토크나이저 (tokenizer.py)
- BPE (Byte Pair Encoding) 구현
- 한국어-영어 혼합 지원
- 특수 토큰 처리 (`<pad>`, `<unk>`, `<bos>`, `<eos>`)

### 2. 모델 아키텍처 (model.py)
- GPT 스타일 디코더 전용 트랜스포머
- 멀티헤드 어텐션 with causal masking
- Position encoding (sin/cos)
- Layer normalization (Pre-LN)
- GELU 활성화 함수

### 3. 학습 (training.py)
- DistributedDataParallel 멀티 GPU 지원
- Mixed Precision (FP16) 학습
- Gradient accumulation
- 코사인 어닐링 스케줄러 with 워밍업
- Gradient clipping

### 4. 추론 (inference.py)
- Top-k, Top-p (nucleus) 샘플링
- Greedy 디코딩
- 반복 패널티
- 대화형 모드
- 성능 벤치마크

## 🎯 사용 예시

### 기본 텍스트 생성

```python
from korean_sllm.inference import TextGenerator

generator = TextGenerator(
    model_path="outputs/best_model.pt",
    tokenizer_path="tokenizer"
)

# 텍스트 생성
generated = generator.generate(
    prompt="한국어는",
    max_length=100,
    temperature=0.8,
    top_p=0.9
)

print(generated[0])
```

### 모델 검증

```python
from korean_sllm.validation import ModelValidator

validator = ModelValidator(
    model_path="outputs/best_model.pt",
    tokenizer_path="tokenizer"
)

# 종합 평가
results = validator.run_comprehensive_evaluation()
print(f"Perplexity: {results['perplexity']:.2f}")
```

## 📈 모니터링

Weights & Biases를 통한 실시간 모니터링:

- 학습/검증 손실
- Perplexity
- 정확도
- 학습률 스케줄
- GPU 메모리 사용량
- 생성 샘플

## 🛠️ 트러블슈팅

### CUDA 메모리 부족
```bash
# 배치 크기나 시퀀스 길이 줄이기
# configs/small_config.json 사용
python common/scripts/train.py --config configs/small_config.json
```

### 멀티 GPU 문제
```bash
# NCCL 환경 변수 설정
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
```

### 데이터셋 다운로드 실패
```bash
# 새로운 데이터셋 다운로드 강제 실행
python common/scripts/train.py --download_fresh

# 또는 전용 다운로드 스크립트 사용
./common/scripts/download_datasets.sh --force
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- Hugging Face Transformers 팀
- OpenAI GPT 논문 저자들
- PyTorch 개발팀
- 데이터 제공: AI Hub, 위키피디아, CommonCrawl

## 📞 연락처

질문이나 제안사항이 있으시면 이슈를 열어주세요.

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!
