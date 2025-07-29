# 한국어 소형 언어모델 추론 모듈

이 모듈은 학습된 한국어 소형 언어모델(Korean SLLM)의 추론을 위한 완전한 솔루션을 제공합니다.

## 🚀 주요 기능

- **모델 로딩**: PyTorch 체크포인트에서 모델 자동 로드
- **텍스트 생성**: 프롬프트 기반 자동 텍스트 생성
- **대화형 채팅**: 컨텍스트를 고려한 대화형 응답 생성
- **텍스트 완성**: 불완전한 텍스트 자동 완성
- **성능 벤치마크**: 모델 성능 측정 및 분석
- **다양한 샘플링**: Temperature, Top-k, Top-p 샘플링 지원

## 📁 파일 구조

```
inference/
├── __init__.py           # 패키지 초기화
├── model.py              # 추론용 모델 클래스
├── tokenizer.py          # 간단한 토크나이저
├── inference_engine.py   # 통합 추론 엔진
├── console_app.py        # 대화형 콘솔 애플리케이션
├── requirements.txt      # 의존성 패키지
├── README.md            # 사용법 안내
└── run_inference.py     # 실행 스크립트
```

## 🛠️ 설치 및 설정

### 1. 의존성 패키지 설치

```bash
pip install -r inference/requirements.txt
```

### 2. 체크포인트 준비

기본적으로 `./outputs/checkpoint-12000/` 경로의 체크포인트를 사용합니다.
체크포인트 폴더에는 다음 파일들이 있어야 합니다:

- `pytorch_model.bin`: 모델 가중치 파일
- `config.json`: 모델 설정 파일

## 🎮 사용 방법

### 1. 대화형 콘솔 실행

가장 간단한 방법은 대화형 콘솔을 실행하는 것입니다:

```bash
# 기본 체크포인트 사용
python -m inference.console_app

# 특정 체크포인트 지정
python -m inference.console_app --checkpoint ./outputs/checkpoint-8000

# CPU 사용 강제
python -m inference.console_app --device cpu

# 토크나이저 파일 지정
python -m inference.console_app --tokenizer ./path/to/vocab.txt
```

### 2. 실행 스크립트 사용

```bash
# 기본 실행
python inference/run_inference.py

# 옵션 지정 실행
python inference/run_inference.py --checkpoint ./outputs/checkpoint-10000 --device cuda
```

## 💬 콘솔 애플리케이션 사용법

### 기본 대화

```
🧑 사용자: 안녕하세요!
🤖 모델: 안녕하세요! 무엇을 도와드릴까요?
```

### 명령어 목록

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `/help` | 도움말 표시 | `/help` |
| `/info` | 모델 정보 표시 | `/info` |
| `/settings` | 현재 설정 확인 | `/settings` |
| `/temp <값>` | 온도 설정 (0.1-2.0) | `/temp 1.2` |
| `/length <값>` | 최대 생성 길이 설정 | `/length 150` |
| `/clear` | 대화 히스토리 초기화 | `/clear` |
| `/save <파일명>` | 히스토리 저장 | `/save my_chat.json` |
| `/load <파일명>` | 히스토리 불러오기 | `/load my_chat.json` |
| `/benchmark` | 성능 테스트 실행 | `/benchmark` |
| `/complete` | 텍스트 완성 모드 | `/complete` |
| `/multiple <n>` | 다중 응답 생성 | `/multiple 3` |
| `/exit` 또는 `/quit` | 프로그램 종료 | `/exit` |

### 텍스트 완성 모드

```
🧑 사용자: /complete

📝 텍스트 완성 모드 (빈 줄 입력 시 종료)
완성할 텍스트: 오늘 날씨가

✨ 완성된 텍스트:
오늘 날씨가 매우 좋습니다. 맑은 하늘과 따뜻한 햇살이...
```

### 다중 응답 생성

```
🧑 사용자: /multiple 3

🔀 다중 응답 모드 (3개 응답 생성)
프롬프트: 한국의 전통 음식은

✨ 3개의 응답:
[응답 1] 김치, 불고기, 비빔밥 등이 있습니다.
[응답 2] 매우 다양하며 지역마다 특색이 있습니다.
[응답 3] 발효 음식이 많고 건강에 좋습니다.
```

## 🔧 프로그래밍 방식 사용

### 기본 사용법

```python
from inference import InferenceEngine

# 모델 로드
engine = InferenceEngine.from_checkpoint(
    checkpoint_path="./outputs/checkpoint-12000",
    device="cuda"
)

# 텍스트 생성
response = engine.generate_text(
    prompt="안녕하세요",
    max_length=100,
    temperature=0.9
)
print(response)
```

### 고급 사용법

```python
from inference import InferenceModel, SimpleTokenizer, InferenceEngine

# 개별 컴포넌트 로드
model = InferenceModel.from_pretrained("./outputs/checkpoint-12000")
tokenizer = SimpleTokenizer("./path/to/vocab.txt")
engine = InferenceEngine(model, tokenizer)

# 채팅 모드
chat_history = []
response = engine.chat_generate(
    message="오늘 날씨 어때?",
    chat_history=chat_history,
    temperature=0.8
)

# 히스토리 업데이트
chat_history.append({"role": "user", "content": "오늘 날씨 어때?"})
chat_history.append({"role": "assistant", "content": response})

# 텍스트 완성
completed = engine.complete_text(
    incomplete_text="인공지능의 미래는",
    max_completion_length=50
)

# 성능 벤치마크
results = engine.benchmark(num_runs=10)
print(f"평균 속도: {results['average_tokens_per_second']:.1f} 토큰/초")
```

## ⚙️ 설정 옵션

### 생성 파라미터

- **`max_length`**: 최대 생성 길이 (기본값: 100)
- **`temperature`**: 샘플링 온도 (기본값: 1.0)
  - 낮을수록 일관된 출력, 높을수록 창의적 출력
- **`top_k`**: Top-k 샘플링 (기본값: 50)
- **`top_p`**: Nucleus 샘플링 (기본값: 0.9)
- **`do_sample`**: 샘플링 여부 (기본값: True)

### 모델 설정

체크포인트의 `config.json`에서 자동으로 로드됩니다:

```json
{
  "vocab_size": 65536,
  "hidden_size": 2048,
  "num_layers": 24,
  "num_heads": 32,
  "intermediate_size": 8192,
  "max_position_embeddings": 4096
}
```

## 📊 성능 최적화

### GPU 메모리 최적화

- 긴 시퀀스보다는 짧은 시퀀스를 여러 번 생성
- 배치 크기를 적절히 조정
- 필요하지 않을 때는 `gradient_checkpointing` 비활성화

### 속도 최적화

- CUDA 사용 시 `torch.compile()` 활용 (PyTorch 2.0+)
- Half precision (FP16) 사용
- Flash Attention 등 최적화된 attention 구현 사용

## 🐛 문제 해결

### 자주 발생하는 오류

1. **모델 로딩 실패**
   ```
   FileNotFoundError: 모델 파일을 찾을 수 없습니다
   ```
   → 체크포인트 경로와 파일 존재 여부 확인

2. **CUDA 메모리 부족**
   ```
   RuntimeError: CUDA out of memory
   ```
   → `--device cpu` 옵션 사용하거나 배치 크기 감소

3. **토크나이저 오류**
   ```
   어휘 파일 로드 실패
   ```
   → 기본 어휘가 자동으로 사용됨 (정상 동작)

### 디버깅 팁

- 로그 출력을 통해 모델 로딩 과정 확인
- 간단한 프롬프트로 먼저 테스트
- GPU 메모리 사용량 모니터링 (`nvidia-smi`)

## 📈 확장 및 개선

### 토크나이저 개선

현재는 간단한 문자 단위 토크나이저를 사용합니다. 더 나은 성능을 위해서는:

- SentencePiece 토크나이저 사용
- BPE (Byte Pair Encoding) 적용
- 서브워드 토크나이징

### 모델 최적화

- Quantization (INT8, INT4)
- Model pruning
- Knowledge distillation

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

---

더 자세한 정보는 각 모듈의 docstring을 참조하세요. 