# 🚀 한국어 SLLM 추론 빠른 시작 가이드

## 📋 목차
- [요구사항](#요구사항)
- [설치](#설치)
- [사용법](#사용법)
- [예시](#예시)
- [문제해결](#문제해결)

## 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (선택적, CPU도 가능)
- 8GB 이상 RAM (GPU 사용 시 4GB 이상 VRAM)

## 설치

1. **의존성 패키지 설치**:
   ```bash
   pip install -r requirements.txt
   ```

2. **모델 체크포인트 준비**:
   - 학습된 모델 체크포인트가 `./outputs/` 디렉토리에 있어야 합니다
   - 체크포인트 폴더에는 `pytorch_model.bin`과 `config.json`이 포함되어야 합니다

## 사용법

### 🎯 방법 1: 간단한 Bash 스크립트 (추천)

```bash
# 사용 가능한 모델 목록 확인
./run.sh

# 특정 모델로 추론 실행
./run.sh ./outputs/checkpoint-12000

# CPU에서 실행
./run.sh ./outputs/checkpoint-12000 cpu

# 도움말 보기
./run.sh --help
```

### 🐍 방법 2: Python 스크립트

```bash
# 사용 가능한 모델 목록 확인
python3 start_inference.py --list

# 특정 모델로 추론 실행
python3 start_inference.py --model ./outputs/checkpoint-12000

# 추가 옵션과 함께 실행
python3 start_inference.py --model ./outputs/checkpoint-12000 --device cpu --tokenizer ./tokenizer.json
```

### 🔧 방법 3: 고급 사용자용 (기존 스크립트)

```bash
# 대화형 모드
python3 run_inference.py

# 특정 체크포인트 지정
python3 run_inference.py --checkpoint ./outputs/checkpoint-12000

# 빠른 테스트
python3 run_inference.py --test --checkpoint ./outputs/checkpoint-8000
```

## 예시

### 기본 대화 예시

```
🇰🇷 한국어 소형 언어모델 (Korean SLLM) 추론 시스템
======================================================================

✅ 모델 디렉토리: ./outputs/checkpoint-12000
🖥️  디바이스: auto

🚀 추론 시스템을 시작합니다...

🤖 한국어 소형 언어모델 대화형 콘솔
============================================================

📊 모델 정보:
  • 모델명: KoreanSLLM
  • 파라미터 수: 1,234,567,890
  • 어휘 크기: 65,536
  • 디바이스: cuda:0
  • 최대 시퀀스 길이: 4096

💬 대화를 시작하세요! (/help 명령어로 도움말 확인)

사용자: 안녕하세요!
모델: 안녕하세요! 무엇을 도와드릴까요?

사용자: /exit
👋 안녕히 가세요!
```

### 명령어 옵션

대화 중 사용할 수 있는 명령어:
- `/help` - 도움말 보기
- `/settings` - 생성 설정 변경
- `/history` - 대화 히스토리 보기
- `/clear` - 대화 히스토리 클리어
- `/save` - 대화 세션 저장
- `/exit` - 프로그램 종료

## 문제해결

### 일반적인 문제들

1. **"모델 파일을 찾을 수 없습니다" 오류**
   ```bash
   # 체크포인트 디렉토리 확인
   ls -la ./outputs/
   
   # 필수 파일 확인
   ls -la ./outputs/checkpoint-12000/
   ```

2. **CUDA 관련 오류**
   ```bash
   # CPU로 강제 실행
   ./run.sh ./outputs/checkpoint-12000 cpu
   ```

3. **메모리 부족 오류**
   - 더 작은 배치 크기 사용
   - CPU 모드로 실행
   - 다른 프로그램 종료 후 재시도

4. **패키지 설치 오류**
   ```bash
   # 가상환경 사용 권장
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### 로그 및 디버깅

- 자세한 오류 정보가 필요한 경우 `--test` 옵션 사용
- 문제 발생 시 전체 스택 트레이스가 출력됩니다

### 성능 최적화

- **GPU 사용**: CUDA가 설치된 경우 자동으로 GPU 사용
- **메모리 효율성**: 큰 모델의 경우 `torch.cuda.empty_cache()` 호출
- **배치 크기**: 메모리에 맞게 조절

## 📞 지원

문제가 지속되는 경우:
1. 오류 메시지와 함께 이슈 제출
2. 시스템 환경 정보 포함 (Python 버전, CUDA 버전 등)
3. 사용한 명령어와 모델 경로 포함

---

🎉 **즐거운 한국어 AI 대화를 즐기세요!** 🇰🇷 