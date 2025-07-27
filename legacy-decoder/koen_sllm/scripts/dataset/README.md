# 한국어 sLLM 데이터셋 처리 모듈

이 디렉토리는 한국어 sLLM을 위한 데이터셋 다운로드와 전처리를 담당하는 모듈들을 포함합니다.

## 📦 모듈 구조

### 🚀 메인 다운로드 스크립트
- **`process_datasets.sh`** - 원시 데이터 다운로드 조정기
  - **순수 다운로드만 수행**
  - 한국어/영어/명령어 데이터 선택 다운로드
  - 테스트용 소량 다운로드 지원

### 📥 다운로드 모듈들 (순수 다운로드만)
- **`download_raw_data.py`** - 원시 데이터 다운로드
  - 한국어/영어 텍스트 데이터
  - 명령어-응답 데이터
  - 메타데이터 자동 생성

### 🔧 전처리 모듈들 (별도 실행)
- **`preprocess_pretraining.py`** - 사전학습 데이터 전처리
  - 한영 혼합 비율 조정 (기본: 한국어 70%, 영어 30%)
  - 품질 필터링 및 중복 제거
  - 언어 감지 및 정규화

- **`preprocess_finetuning.py`** - 미세조정 데이터 전처리
  - **최소 5만개 이상 보장**
  - 데이터 증강 기법 적용
  - 태스크별 분류 및 균형 조정

### 🛠️ 지원 모듈들
- **`download_core.sh`** - 공통 유틸리티 함수들
- **기존 스크립트들** - 호환성을 위해 유지

## 🎯 주요 특징

### ✅ 완전한 역할 분리
```
다운로드 스크립트 → 원시 데이터만 다운로드
전처리 스크립트 → 용도별 데이터 가공 (별도 실행)
```

### 🌍 한영 혼합 지원
- **사전학습**: 한국어 70% + 영어 30%
- **미세조정**: 한국어 60% + 영어 40%
- 언어별 비율 조정 가능

### 📊 품질 보장
- **미세조정 데이터**: 최소 5만개 이상
- **품질 필터링**: 길이, 반복 패턴, 의미성 검사
- **중복 제거**: 해시 기반 중복 데이터 제거
- **데이터 증강**: 부족한 데이터 자동 보완

## 🚀 빠른 시작

### 1단계: 원시 데이터 다운로드
```bash
# 모든 원시 데이터 다운로드
./process_datasets.sh

# 한국어 데이터만
./process_datasets.sh --korean

# 테스트용 소량 데이터
./process_datasets.sh --small

# 명령어 데이터만
./process_datasets.sh --instructions
```

### 2단계: 전처리 (선택사항)
```bash
# 사전학습 데이터 전처리
python3 preprocess_pretraining.py --raw-data-dir raw_datasets

# 미세조정 데이터 전처리 (5만개 이상)
python3 preprocess_finetuning.py --raw-data-dir raw_datasets

# 한영 혼합 사전학습 데이터
python3 preprocess_pretraining.py --mixed-only

# 대규모 미세조정 데이터 (10만개 목표)
python3 preprocess_finetuning.py --target-count 100000
```

## 📁 생성되는 데이터 구조

### 다운로드 후 (1단계)
```
raw_datasets/
├── korean_raw_*.jsonl                 # 한국어 원시 데이터
├── english_raw_*.jsonl                # 영어 원시 데이터
├── instruction_raw_*.jsonl            # 명령어 원시 데이터
└── download_metadata.json             # 다운로드 메타데이터
```

### 전처리 후 (2단계)
```
datasets/
├── mixed_pretraining_corpus.json      # 한영 혼합 사전학습 데이터
├── mixed_instructions.json            # 한영 혼합 미세조정 데이터 (5만개+)
├── mixed_task_distribution.json       # 태스크별 분포 정보
├── pretraining_preprocessing_stats.json
├── finetuning_preprocessing_stats.json
└── (통계 및 메타데이터 파일들...)
```

## ⚙️ 사용 패턴

### 🎯 권장 사용법 (단계별)
```bash
# 1단계: 원시 데이터 다운로드
./process_datasets.sh --all

# 2단계: 필요에 따라 전처리
python3 preprocess_pretraining.py    # 사전학습용
python3 preprocess_finetuning.py     # 미세조정용 (5만개+)
```

### ⚡ 빠른 테스트
```bash
# 소량 테스트 데이터로 전체 워크플로우 확인
./process_datasets.sh --small
python3 preprocess_pretraining.py --raw-data-dir raw_datasets --mixed-only
python3 preprocess_finetuning.py --raw-data-dir raw_datasets --target-count 5000
```

### 🎨 커스텀 설정
```bash
# 한국어만 다운로드 + 전처리
./process_datasets.sh --korean
python3 preprocess_pretraining.py --korean-only
python3 preprocess_finetuning.py --korean-only

# 대용량 미세조정 데이터 생성
./process_datasets.sh --instructions
python3 preprocess_finetuning.py --target-count 200000
```

## 🔍 검증 및 확인

### 다운로드 확인
```bash
# 다운로드된 파일 확인
ls -la raw_datasets/

# 메타데이터 확인
cat raw_datasets/download_metadata.json | python3 -m json.tool
```

### 전처리 결과 확인
```bash
# 미세조정 데이터 개수 확인
python3 -c "
import json
with open('datasets/mixed_instructions.json', 'r') as f:
    data = json.load(f)
print(f'미세조정 데이터: {len(data)}개')
"

# 언어 분포 확인
cat datasets/finetuning_preprocessing_stats.json | python3 -m json.tool

# 태스크 분포 확인
cat datasets/mixed_task_distribution.json | python3 -m json.tool
```

## ⚙️ 고급 옵션

### 언어 비율 조정
```bash
# 사전학습: 한국어 80%, 영어 20%
python3 preprocess_pretraining.py --korean-ratio 0.8 --english-ratio 0.2

# 미세조정: 한국어 70%, 영어 30%
python3 preprocess_finetuning.py --korean-ratio 0.7 --english-ratio 0.3
```

### 목표 데이터 크기 설정
```bash
# 미세조정 데이터 15만개 목표
python3 preprocess_finetuning.py --target-count 150000

# 최소 8만개 보장
python3 preprocess_finetuning.py --min-target 80000
```

### 데이터 증강 제어
```bash
# 데이터 증강 비활성화
python3 preprocess_finetuning.py --no-augmentation
```

## 🐛 트러블슈팅

### 1. 다운로드 실패
```bash
# Python 패키지 설치
pip install datasets tqdm

# 디스크 공간 확인
./process_datasets.sh --check

# 소량 데이터로 테스트
./process_datasets.sh --small
```

### 2. 미세조정 데이터가 5만개 미달
```bash
# 모든 명령어 데이터 다운로드
./process_datasets.sh --instructions

# 데이터 증강 활성화로 재실행
python3 preprocess_finetuning.py --target-count 80000
```

### 3. 메모리 부족
```bash
# 작은 배치로 처리
python3 preprocess_finetuning.py --target-count 30000
```

## 📊 성능 최적화

### 병렬 처리
```bash
# 언어별로 병렬 다운로드
./process_datasets.sh --korean &
./process_datasets.sh --english &
wait

# 그 다음 전처리
python3 preprocess_pretraining.py
python3 preprocess_finetuning.py
```

### 캐시 활용
- 원시 데이터는 한 번 다운로드 후 재사용
- `--force` 옵션으로 강제 재다운로드 가능

## 💡 워크플로우 팁

1. **처음 사용**: `--small` 옵션으로 테스트 후 전체 실행
2. **개발용**: 다운로드와 전처리를 분리하여 개발
3. **운영용**: 전체 다운로드 후 필요한 전처리만 선택 실행
4. **품질 검증**: 생성된 통계 파일들을 반드시 확인

## 📞 문의

데이터셋 관련 문제나 개선 사항이 있으시면 이슈를 등록해 주세요. 