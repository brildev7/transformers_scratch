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
├── dataset/           # 데이터셋 설정 (⭐ 업데이트됨)
│   ├── pretraining.json           # 사전훈련용 데이터 설정 (v2.0)
│   ├── finetuning.json            # 파인튜닝용 데이터 설정
│   ├── korean_datasets.json       # 한국어 데이터셋 (허용적 라이선스 중심)
│   ├── english_datasets.json      # 영어 데이터셋 (허용적 라이선스 중심)
│   ├── recipe1_commercial_ready.json  # 레시피 1: 상용화 대비
│   ├── recipe2_performance_max.json   # 레시피 2: 성능 극대화
│   ├── korean_instructions.json   # 한국어 명령어 데이터
│   └── english_instructions.json  # 영어 명령어 데이터
└── README.md           # 이 파일
```

## 🎯 데이터셋 전략 (RecommendDataset.md 기반)

### **레시피 1: 상용화 대비 및 민첩성 중심** ⚖️ **[균형잡힌 이중언어]**
- **파일**: `recipe1_commercial_ready.json`
- **총 규모**: ~75GB (한국어 37.5GB + 영어 37.5GB)
- **혼합 비율**: 한국어 50%, 영어 50%
- **라이선스**: 완전 허용적 (Apache 2.0, CC0, ODC-BY 등)
- **장점**: 법적 리스크 최소화, 미래 상업화 자유, 균형잡힌 이중언어 능력
- **대상**: 소규모 연구소, 스타트업, 상업적 활용 계획

### **레시피 2: 성능 극대화** ⚖️ **[균형잡힌 이중언어]**
- **파일**: `recipe2_performance_max.json`
- **총 규모**: ~150GB (한국어 75GB + 영어 75GB)
- **혼합 비율**: 한국어 50%, 영어 50%
- **라이선스**: 혼합형 (AI-Hub 포함으로 상업적 제약)
- **장점**: 최대 데이터 규모, 벤치마크 성능 극대화, 균형잡힌 이중언어 능력
- **대상**: 순수 학술 연구, 성능 검증 목적

### **레시피 3: 한국어 중심 학습** 🇰🇷 **[한국어 특화]** 
- **파일**: `recipe3_korean_focused.json`
- **총 규모**: ~100GB (한국어 70GB + 영어 30GB)
- **혼합 비율**: 한국어 70%, 영어 30%
- **라이선스**: 혼합형 (AI-Hub 포함)
- **장점**: 한국어 능력 극대화, 한국어 특화 최적화
- **대상**: 한국어 중심 AI 서비스, 한국어 교육/연구

## 🔧 데이터셋 구성 요소

### 한국어 데이터 (v2.0 업데이트)
| 데이터셋 | 크기 | 라이선스 | 품질 | 레시피1 | 레시피2 | 레시피3 |
|---------|------|----------|------|---------|---------|---------|
| lcw99/wikipedia-korean | ~2GB | Apache-2.0 | 높음 | ✅ | ✅ | ✅ |
| kowikitext | ~1.7GB | CC-BY-SA-3.0 | 높음 | ✅ | ✅ | ✅ |
| korean_common_crawl | ~33-55GB | CC-BY | 높음 | ✅ | ✅ | ✅ |
| korean_news_data | ~0.5GB | 연구친화 | 중간 | ✅ | ✅ | ✅ |
| AI-Hub 초거대AI 데이터 | ~16GB | 연구제한 | 높음 | ❌ | ✅ | ✅ |

### 영어 데이터 (v2.0 업데이트)
| 데이터셋 | 크기 | 라이선스 | 품질 | 레시피1 | 레시피2 | 레시피3 |
|---------|------|----------|------|---------|---------|---------|
| C4 realnewslike | 10-15GB | ODC-BY | 높음 | ✅ | ✅ | ✅ |
| OpenWebText | 15GB | CC0 | 높음 | ✅ | ❌ | ❌ |
| BookCorpus | 5GB | 연구커뮤니티 | 높음 | ✅ | ✅ | ✅ |
| The Pile/Gutenberg | 2.5-9.5GB | Public Domain | 높음 | ✅ | ❌ | ✅ |
| The Pile/arXiv | 30GB | CC-BY | 높음 | ❌ | ✅ | ❌ |
| The Pile/OpenWebText2 | 30GB | 허용적 | 높음 | ❌ | ✅ | ❌ |
| WikiText-103 | 0.5GB | CC-BY-SA | 높음 | ❌ | ✅ | ✅ |

### ⚠️ 제외된 데이터셋
- **The Pile/Books3**: 저작권 침해 위험 (DMCA 대상)
- **AI-Hub 데이터** (레시피1): 상업적 활용 시 별도 협의 필요
- **RedPajama**: OpenWebText가 더 안정적

## 🚀 사용 방법

### 레시피 기반 데이터셋 구성
```bash
# 레시피 1: 상용화 대비 (균형잡힌 50:50)
python preprocess_pretraining.py --recipe configs/dataset/recipe1_commercial_ready.json

# 레시피 2: 성능 극대화 (균형잡힌 50:50)
python preprocess_pretraining.py --recipe configs/dataset/recipe2_performance_max.json

# 레시피 3: 한국어 중심 (한국어 70:30)
python preprocess_pretraining.py --recipe configs/dataset/recipe3_korean_focused.json

# 기본 설정 사용 (한국어 70:30)
python preprocess_pretraining.py --config configs/dataset/pretraining.json
```

### 개별 언어 데이터셋 구성
```bash
# 한국어만
python preprocess_pretraining.py --korean-config configs/dataset/korean_datasets.json

# 영어만  
python preprocess_pretraining.py --english-config configs/dataset/english_datasets.json
```

## ⚙️ 라이선스 관리

### 허용적 라이선스 (상업적 이용 가능)
- **Apache 2.0**: 위키피디아 한국어
- **CC0**: OpenWebText (Public Domain에 가까움)
- **ODC-BY**: C4 데이터셋
- **Public Domain**: Gutenberg 고서

### 제한적 라이선스
- **CC-BY-SA**: 동일 라이선스 유지 조건
- **연구제한**: AI-Hub (상업적 이용 시 별도 협의)

### 라이선스 추적
```json
{
  "license_tracking": {
    "enabled": true,
    "output_file": "dataset_license_manifest.json",
    "description": "연구 재현성 및 향후 실사 대비"
  }
}
```

## 🔧 전처리 파이프라인

### 필수 전처리 과정
1. **중복 제거 (Deduplication)**
   - 방법: LSH (Local-Sensitivity Hashing)
   - 출처: OpenWebText 구축 기법

2. **품질 필터링 (Quality Filtering)**
   - 방법: C4 데이터셋 정제 과정 응용
   - 기준: 최소 문장 수, 반복 비율, HTML 태그 제거

3. **통합 토크나이저 구축**
   - 방법: SentencePiece BPE
   - 어휘집 크기: 50K-100K (권장: 65K)
   - 이중 언어 지원

## 📊 권장 설정별 계산 요구사항

| 레시피 | 데이터 크기 | 언어 비율 | 예상 계산 비용 | 메모리 요구사항 | 적합한 GPU |
|--------|------------|-----------|----------------|----------------|------------|
| 레시피 1 | 75GB | 50:50 | 수천 GPU-hour | 16-32GB | V100, A100 |
| 레시피 2 | 150GB | 50:50 | 수만 GPU-hour | 32-80GB | A100, H100 |
| 레시피 3 | 100GB | 70:30 | 수천-수만 GPU-hour | 24-48GB | A100, H100 |

## 🎯 전략적 의사결정 가이드

### 소규모 연구소/스타트업 → **레시피 1** ⚖️
- ✅ 법적 리스크 최소화
- ✅ 미래 상업화 옵션 확보
- ✅ 균형잡힌 이중언어 능력
- ✅ 적당한 계산 비용

### 대학/연구기관 (순수 연구) → **레시피 2** ⚖️
- ✅ 최대 성능 달성
- ✅ 벤치마크 경쟁력
- ✅ 균형잡힌 이중언어 능력
- ⚠️ 상업적 활용 제약

### 한국어 중심 서비스 → **레시피 3** 🇰🇷
- ✅ 한국어 능력 극대화
- ✅ 한국어 특화 최적화
- ✅ 한국 문화/언어 맥락 보존
- ⚠️ 영어 능력은 보조적

## 🔗 관련 문서

- [RecommendDataset.md](../reference/RecommendDataset.md) - 데이터셋 분석 및 권고사항
- [Training Guide](../docs/training.md)
- [Model Architecture](../docs/models.md)
- [Dataset Preparation](../common/scripts/dataset/README.md)
- [License Compliance](../docs/license_compliance.md)

## 📝 버전 히스토리

### v2.0 (현재)
- RecommendDataset.md 권고사항 반영
- 허용적 라이선스 중심 재구성
- 레시피 1, 2 추가
- 라이선스 정보 명시
- AI-Hub 데이터 제외 (레시피 1)

### v1.1 (이전)
- 기본 데이터셋 설정
- 한국어/영어 50:50 비율 

## 📋 주요 업데이트 내용

### 1. **한국어 데이터셋 설정** (`korean_datasets.json`)
- ✅ **허용적 라이선스 중심**으로 재구성
- ✅ `lcw99/wikipedia-korean` (Apache 2.0) 우선 순위
- ✅ `kowikitext` (CC-BY-SA 3.0) 포함  
- ❌ AI-Hub 데이터 제외 (상업적 제약)
- ✅ 라이선스 정보 상세 명시

### 2. **영어 데이터셋 설정** (`english_datasets.json`)
- ✅ `C4 realnewslike` (15GB, ODC-BY) 최우선
- ✅ `OpenWebText` (40GB, CC0) 추가
- ✅ `BookCorpus` (5GB) 포함
- ✅ The Pile 구성요소 정보 추가
- ❌ Books3 제외 (저작권 위험)

### 3. **사전학습 설정** (`pretraining.json`)
- ✅ 혼합 비율 조정: **한국어 4%, 영어 96%** (레시피 1 기준)
- ✅ 중복 제거 (LSH) 및 품질 필터링 설정
- ✅ 이중 언어 토크나이저 설정 (65K vocab)
- ✅ 라이선스 추적 기능 추가

### 4. **새로운 레시피 설정 파일**

#### 🚀 **레시피 1: 상용화 대비** (`recipe1_commercial_ready.json`)
- **총 75GB** (한국어 3GB + 영어 72GB)
- **완전 허용적 라이선스**만 사용
- **법적 리스크 최소화**
- **소규모 연구소 권장**

#### 🔬 **레시피 2: 성능 극대화** (`recipe2_performance_max.json`)
- **총 150GB** (한국어 18GB + 영어 132GB)  
- AI-Hub 데이터 포함으로 **더 큰 규모**
- **벤치마크 성능 극대화**
- **순수 학술 연구용**

### 5. **README.md 대폭 개선**
- ✅ 레시피별 전략 가이드 추가
- ✅ 라이선스 관리 섹션 신설
- ✅ 데이터셋 비교 표 제공
- ✅ 계산 요구사항 가이드

## 🎯 핵심 권고사항 (RecommendDataset.md 반영)

### **소규모 연구소**에게는 **레시피 1** 강력 권장 ⭐
- 법적 안정성과 미래 상업화 자유 확보
- 데이터 품질 > 데이터 양 전략
- 민첩한 연구 개발 가능

### 주요 제외 데이터
- **AI-Hub**: 상업적 활용 시 별도 협의 필요
- **Books3**: 저작권 침해 위험 (DMCA 대상)

이제 다음과 같은 명령어로 권고사항을 바탕으로 데이터를 전처리할 수 있습니다:

```bash
# 상용화 대비 레시피 (권장)
python preprocess_pretraining.py --recipe configs/dataset/recipe1_commercial_ready.json

# 성능 극대화 레시피
python preprocess_pretraining.py --recipe configs/dataset/recipe2_performance_max.json
``` 