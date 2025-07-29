# 🤖 한국어 sLLM 토크나이저 선택 가이드

이 가이드는 한국어 언어모델 학습 시 사용할 수 있는 다양한 토크나이저 옵션에 대해 설명합니다.

## 📚 토크나이저 종류

### 1️⃣ Gemma3 토크나이저 (`gemma3`) 🏆 **기본값**

**특징:**
- Google의 검증된 고성능 토크나이저
- 다국어 지원 (한국어 포함)
- Subword 기반 토크나이징
- 256,000개 어휘 크기 (대용량)

**장점:**
- Google에서 검증된 고품질 토크나이저
- 우수한 다국어 처리 성능
- 효율적인 서브워드 분할
- 안정적이고 일관된 결과

**단점:**
- 큰 어휘 크기로 인한 메모리 사용량 증가
- 인터넷 연결 필요 (초기 다운로드)

**사용 예:**
```bash
# 기본값이므로 별도 지정 불필요
python3 train_h100_dual.py --test-mode
# 또는 명시적 지정
python3 train_h100_dual.py --test-mode --tokenizer-type gemma3
```

**토큰화 예시:**
```
원문: "안녕하세요, 저는 한국어 언어모델입니다."
토큰: ['▁안녕하세요', ',', '▁저는', '▁한국어', '▁언어', '모델', '입니다', '.']
```

### 2️⃣ 개선된 한국어 토크나이저 (`improved_korean`)

**특징:**
- 공백 기준 토크나이징
- 단어 단위 맥락 보존
- Subword 분할 지원
- 학습 데이터 기반 어휘 구축

**장점:**
- 맥락 정보 보존
- 자연스러운 토큰화
- 확장 가능한 어휘
- 한국어 특성 고려

**단점:**
- 상대적으로 작은 어휘 크기
- 초기 어휘 구축 필요

**사용 예:**
```bash
python3 train_h100_dual.py --tokenizer-type improved_korean
```

**토큰화 예시:**
```
원문: "안녕하세요, 저는 한국어 언어모델입니다."
토큰: ['안녕하세요', ',', '저는', '한국어', '언어모델', '입니다', '.']
```

### 3️⃣ 기본 한국어 토크나이저 (`korean`)

**특징:**
- 형태소 분석 기반 토크나이징
- 조사/어미 자동 분리
- 한국어 교착어 특성 고려
- 구두점 처리

**장점:**
- 한국어 문법 구조 반영
- 세밀한 형태소 단위 분석
- 일관된 토큰 생성

**단점:**
- 공백 정보 손실
- 맥락 학습 어려움
- 복잡한 전처리

**사용 예:**
```bash
python3 train_h100_dual.py --tokenizer-type korean
```

**토큰화 예시:**
```
원문: "안녕하세요, 저는 한국어 언어모델입니다."
토큰: ['안녕하', '세요', ',', '저는', '한국어', '언어모델', '입니다', '.']
```

## 🚀 사용법

### 기본 사용법

```bash
# 1. Gemma3 토크나이저 (기본값)
python3 train_h100_dual.py --test-mode

# 2. 명시적 지정
python3 train_h100_dual.py --test-mode --tokenizer-type gemma3

# 3. 개선된 한국어 토크나이저
python3 train_h100_dual.py --test-mode --tokenizer-type improved_korean

# 4. 기본 한국어 토크나이저
python3 train_h100_dual.py --test-mode --tokenizer-type korean
```

### run_h100_dual.sh 사용법

```bash
# 기본값 (Gemma3)
./run_h100_dual.sh --test

# 다른 토크나이저 선택
./run_h100_dual.sh --test --tokenizer-type improved_korean
./run_h100_dual.sh --test --tokenizer-type korean

# 실제 학습
./run_h100_dual.sh --train --tokenizer-type gemma3
```

### 전체 학습 실행

```bash
# Gemma3 토크나이저로 실제 학습 (추천)
python3 train_h100_dual.py \
    --tokenizer-type gemma3 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --save-steps 500
```

### 기타 옵션과 함께 사용

```bash
# 모든 옵션 조합
python3 train_h100_dual.py \
    --tokenizer-type gemma3 \
    --dataset-path "../../../../datasets" \
    --output-dir "./outputs" \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-seq-length 2048 \
    --mixed-precision bf16 \
    --save-steps 500
```

## 📊 성능 비교

| 토크나이저 | 어휘 크기 | 맥락 보존 | 다국어 지원 | 성능 | 추천도 |
|-----------|---------|----------|------------|------|--------|
| **gemma3** | 256,000 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🏆 **최고** |
| improved_korean | 32,000 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🥈 좋음 |
| korean | 65,536 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 🥉 보통 |

## 🎯 추천 시나리오

### 🏆 일반적인 경우 → **Gemma3 토크나이저**
```bash
./run_h100_dual.sh --train --tokenizer-type gemma3
```
- 가장 안정적이고 검증된 선택
- 우수한 다국어 성능
- Google에서 지속적으로 업데이트

### 🇰🇷 한국어 특화 → **개선된 한국어 토크나이저**
```bash
./run_h100_dual.sh --train --tokenizer-type improved_korean
```
- 한국어 특성에 최적화
- 맥락 보존 우수
- 커스터마이징 가능

### 🔬 실험/연구 → **기본 한국어 토크나이저**
```bash
./run_h100_dual.sh --train --tokenizer-type korean
```
- 형태소 분석 실험
- 전통적인 한국어 처리 방식
- 세밀한 분석 필요 시

## 🔧 트러블슈팅

### Gemma3 토크나이저 오류

**문제:** `ImportError: No module named 'transformers'`
```bash
# 해결방법
pip install transformers>=4.35.0
```

**문제:** 인터넷 연결 오류
```bash
# 오프라인 환경에서는 다른 토크나이저 사용
./run_h100_dual.sh --train --tokenizer-type improved_korean
```

**문제:** CUDA 메모리 부족
```bash
# 배치 크기 줄이기
./run_h100_dual.sh --train --tokenizer-type gemma3 --batch-size 2
```

### 일반적인 문제

**문제:** 토크나이저 파일 없음
```bash
# 파일 존재 확인
ls -la korean_tokenizer.py improved_korean_tokenizer.py gemma3_tokenizer.py
```

**문제:** 메모리 부족
```bash
# 시퀀스 길이 줄이기
./run_h100_dual.sh --train --max-seq-length 1024
```

## 🚀 향후 개발 계획

1. **성능 최적화**
   - Gemma3 토크나이저 메모리 효율성 개선
   - 동적 어휘 크기 조정

2. **새로운 토크나이저 추가**
   - GPT-4 토크나이저 지원
   - 한국어 특화 BERT 토크나이저

3. **호환성 개선**
   - 추론 시스템과의 완전한 호환성
   - 자동 토크나이저 매칭 기능

---

📝 **참고:** 이 가이드는 지속적으로 업데이트됩니다. 최신 정보는 공식 문서를 확인하세요. 