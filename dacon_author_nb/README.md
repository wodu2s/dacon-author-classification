# 소설 작가 분류 — 나이브 베이즈 · 앙상블 파트

응용머신러닝(2026-2) 기말 팀 프로젝트 · 월간 데이콘 「소설 작가 분류 AI 경진대회」(235670)
영어 소설 문장 → 작가 0~4 다중분류, 평가지표 **multi-class logloss**.

팀 진행: ① 텍스트 전처리 → **② 나이브 베이즈 → ③ 앙상블 (이 폴더, 앙상블은 [8절](#8-앙상블-단계))**

---

## 1. 폴더 구조

```
dacon_author_nb/
├── data/
│   ├── raw/                 # 데이콘 원본을 직접 넣을 자리 (현재 비어 있음)
│   └── processed/           # 전처리 담당이 준 정리본 (train_clean / val_clean / test_clean / sample_submission ...)
├── src/
│   ├── config.py            # ★ 설정은 여기 한 곳: 데이터 경로·열 이름·클래스 수·시드·폴드 수·실험 축
│   ├── data.py              # 데이터 로드/검사, 팀 공통 폴드, 전처리 변형(불용어·문장부호 제거)
│   ├── experiments.py       # 실험 이름 ↔ 설정, 벡터라이저/NB/보정 모델 생성, 단계별 실험 목록
│   ├── cv.py                # 5-fold 실행기: OOF·테스트 확률 생성, 지표 계산, 실험마다 즉시 저장
│   ├── make_folds.py        # 데이터 확인 + outputs/folds.csv 생성
│   ├── run_nb.py            # 실험 실행 (단계별 / 이름 지정)
│   ├── report.py            # 상위 5개 표, 단계별 요약(md), alpha-logloss 그래프
│   ├── make_submission.py   # 최고 설정 → 제출 csv
│   ├── ensemble_common.py   # [앙상블] 기본 모델(LR/SVM/배깅) 생성, 5-fold 실행, 결과 저장 공통 모듈
│   ├── run_base.py          # [앙상블 4-1] 기본 모델: 로지스틱 회귀(단어·글자), 선형 SVM + 확률 보정
│   ├── run_bagging.py       # [앙상블 4-2] 배깅
│   ├── run_voting.py        # [앙상블 4-3] soft voting (단순 / 가중 평균)
│   ├── run_stacking.py      # [앙상블 4-4] 스태킹 (LightGBM / 로지스틱 회귀 메타 모델)
│   ├── report_ensemble.py   # [앙상블 4-5] 방식별 비교표 + 그래프
│   └── make_ensemble_submission.py  # [앙상블 4-6] 최고 방법 → 제출 csv
├── outputs/
│   ├── folds.csv            # ★ 팀 공유: 행 index → fold 번호
│   ├── oof/{실험이름}.npy     # ★ 학습 전체 OOF 확률 (54,728 × 5)
│   ├── test_pred/{실험이름}.npy # ★ 테스트 확률 = 5개 폴드 모델 평균 (19,617 × 5)
│   └── submissions/
│       ├── nb_best.csv                 # ★ 데이콘 제출용 (+ nb_best.json: 어떤 실험인지)
│       ├── nb_best_clip0.0001.csv      # 확률 하한 1e-4 적용본 (7절 참고)
│       └── ensemble_best.csv           # ★ 앙상블 최고 방법 제출용 (+ ensemble_best.json, 8절 참고)
├── results/
│   ├── nb_results.csv       # 실험 결과표 (실험마다 한 줄)
│   ├── nb_summary.md        # 단계별 비교표 (보고서용)
│   ├── alpha_logloss.png    # alpha 에 따른 logloss 그래프 (보고서용)
│   ├── run_full.log, run_extra_calib.log   # 실행 로그
│   ├── ensemble_results.csv # 앙상블 단계 결과표 (실험마다 한 줄)
│   ├── ensemble_summary.md  # 방식별 비교표 (보고서용)
│   ├── ensemble_compare.png # 방식별 최고 logloss 그래프 (보고서용)
│   └── run_ensemble_*.log   # 앙상블 실행 로그
└── requirements.txt
```

## 2. 실행 방법

```bash
pip install -r requirements.txt          # pandas, numpy, scipy, scikit-learn, matplotlib, lightgbm(앙상블)

python src/make_folds.py                 # 1) 데이터 확인(열·행 수·작가 분포) + outputs/folds.csv 생성
python src/run_nb.py --stage all         # 2) 1→2→3→4단계 실험 (이미 끝난 실험은 자동으로 건너뜀)
python src/report.py                     # 3) 상위 5개 표 + results/nb_summary.md + results/alpha_logloss.png
python src/make_submission.py            # 4) outputs/submissions/nb_best.csv
```

- 빠른 점검: `python src/run_nb.py --stage all --sample 2000` → 2,000행 층화 샘플로 전체 파이프라인을 돌려
  `sample_run_2000/` 아래에 따로 저장 (본 결과와 섞이지 않음). `report.py`, `make_submission.py` 에도 `--sample 2000`.
- 실험 하나만: `python src/run_nb.py --exp mnb_count_word12_a0.1` (여러 개 나열 가능, `--force` 로 재실행)
- 특정 실험으로 제출 파일: `python src/make_submission.py --exp <실험이름>` → `outputs/submissions/<실험이름>.csv`
- 폴드는 `--n-jobs`(기본 5) 개 프로세스로 병렬 실행. 전체 실행 시간은 아래 결과 참고.

**실험 이름 규칙**: `{모델}_{벡터라이저}_{단위}{n-gram 하한}{상한}_a{alpha}[_p{전처리}][_cal{보정}]`

| 토큰 | 값 |
|---|---|
| 모델 | `mnb` MultinomialNB(빈도) / `bnb` BernoulliNB(출현 유무) |
| 벡터라이저 | `count` CountVectorizer / `tfidf` TfidfVectorizer |
| 단위 | `word12` 단어 (1,2)-gram / `charwb24` 글자(char_wb) (2,4)-gram / `char35` 글자(char) … |
| 전처리 | 생략=유지 / `_pnostop` 불용어 제거 / `_pnopunct` 문장부호 제거 / `_pnostopnopunct` 둘 다 |
| 보정 | 생략=없음 / `_calsig` sigmoid(Platt) / `_caliso` isotonic |

### 데이터 교체 (원본 ↔ 전처리본)

`src/config.py` 의 `DATASETS` 에 파일 경로가, `ID_COL / TEXT_COL / LABEL_COL` 에 열 이름이 있다.

- `clean`(기본): `data/processed/` 의 `train_clean.csv + val_clean.csv` 를 합쳐 학습 데이터로, `test_clean.csv` 를 테스트로 사용
- `raw`: 데이콘 원본 `train.csv / test_x.csv / sample_submission.csv` 를 `data/raw/` 에 넣고 `--data raw`
  (결과는 `run_raw/` 에 따로 저장, 폴드도 따로 생성됨)
- 새 전처리본을 받으면 `DATASETS` 에 항목을 하나 추가하고 `--data <이름>` 으로 실행

## 3. 검증 설계 (앙상블 담당과 공유하는 틀)

- **폴드**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — 작가 비율을 폴드마다 동일하게.
  결과를 `outputs/folds.csv` 로 저장 → 앙상블 담당도 이 파일을 그대로 읽어서 써야 OOF 가 맞물린다.
- **누수 방지**: 벡터라이저(어휘·IDF)는 **각 폴드의 학습 부분에만 fit**, 검증 폴드·테스트는 transform 만.
  확률 보정도 벡터라이저를 Pipeline 에 넣어, 보정기 학습용 내부 CV 에서도 내부 학습 부분에만 fit.
- **OOF 확률**: 각 행은 그 행을 학습에 쓰지 않은 폴드 모델이 예측한 값 → 학습 데이터 전체에 대해 "본 적 없는 데이터" 예측.
- **테스트 확률**: 5개 폴드 모델의 테스트 예측 평균.
- **지표**: OOF 기준 logloss(주 지표), accuracy, macro-F1. 참고로 `logloss_testlike`(아래 4절) 도 기록.
- 전처리 변형(불용어·문장부호 제거)은 행마다 규칙으로 바꾸는 변환이라 학습 통계를 쓰지 않음 → 분할 전에 적용해도 누수 없음.

## 4. 실험 설계 (단계적으로 좁히기)

| 단계 | 무엇을 | 고정값 |
|---|---|---|
| 1 | 모델(MNB/BNB) × 벡터라이저(Count/TF-IDF) × 단위(단어 (1,1)/(1,2), 글자 char_wb (2,4)/(1,5)) = 12개 | alpha=1 (sklearn 기본 = 고전적 라플라스 add-one) |
| 2 | 1단계 상위 3개 + (모델,벡터라이저) 조합별 1위 → alpha ∈ {0.01, 0.03, 0.1, 0.3, 1} | |
| 3 | 2단계 최고 설정 + 다른 단위(단어↔글자) 최고 설정 → 전처리 4종 (유지 / 불용어 제거 / 문장부호 제거 / 둘 다) | 해당 설정의 alpha |
| 4 | 보정 전 logloss 최고 설정(+ 정확도 최고 설정이 다르면 그것도) → `CalibratedClassifierCV` sigmoid / isotonic | 내부 5-fold, ensemble=False |
| 보충 | 과신이 심한 Count 기반 모델에도 보정 (보정 효과 시연용) | |

보충 실험 재현: `python src/run_nb.py --exp mnb_count_word12_a1_calsig mnb_count_word12_a1_caliso bnb_count_word11_a0.03_calsig bnb_count_word11_a0.03_caliso`

설계 메모

- **BernoulliNB × TF-IDF 는 생략**: BNB 는 `binarize=0` 으로 값을 0/1 로 바꾸므로 Count 와 결과가 똑같다.
- **단어 토큰에 문장부호 포함**: sklearn 기본 토큰 규칙은 문장부호와 1글자 단어("I", "a")를 버리는데,
  작가마다 따옴표 모양(“ ” ‘ ’ vs ")·쌍반점·대시 습관이 달라 이것도 문체 신호다 → `config.WORD_TOKEN_PATTERN` 으로 포함.
- 공통 벡터라이저 설정: `lowercase=True`, `min_df=2`, TF-IDF 는 `sublinear_tf=True` (전처리 담당과 동일).
- 2단계에서 조합별 1위도 넣는 이유: alpha=1 은 TF-IDF 값(0~1 사이)에 비해 스무딩이 과해 TF-IDF 가 1단계에서 불리하기 때문.
- **테스트 문장이 학습보다 약 2배 길다** (평균 487자 vs 228자, 테스트 최소 131자). NB 는 단어마다 로그확률을 더하므로
  문장이 길수록 확률이 더 극단적이 된다 → OOF logloss 가 실제 테스트 logloss 와 다를 수 있다.
  그래서 테스트 최소 길이 이상인 학습 행만으로 계산한 `logloss_testlike` 를 참고 지표로 함께 기록했다.

## 5. 결과 파일

`results/nb_results.csv` 열 설명

| 열 | 의미 |
|---|---|
| exp | 실험 이름 (= oof/test_pred 파일 이름) |
| model / vectorizer / analyzer / ngram | 모델, 벡터라이저, n-gram 단위(word/char_wb), 범위 |
| alpha | 라플라스 스무딩 강도 |
| calibration | 보정 여부 (none / sigmoid / isotonic) |
| prep | 전처리 버전 (keep / nostop / nopunct / nostopnopunct) |
| data_version | 데이터 버전 (clean = 전처리 담당 정리본) |
| logloss | **OOF logloss (주 지표)** |
| logloss_fold_std | 폴드별 logloss 의 표준편차 |
| logloss_testlike | 테스트 최소 길이(131자) 이상인 학습 행만의 OOF logloss (참고) |
| accuracy / macro_f1 | OOF 정확도 / Macro-F1 |
| n_features | 폴드 평균 특징(어휘) 수 |
| time_sec | 소요시간(초): 5개 폴드의 벡터화+학습+예측 시간 합 (단독 실행 기준, 병렬 실행이면 실제 대기시간은 더 짧음) |

## 6. 앙상블 담당에게 넘길 파일

| 파일 | 모양 | 행 순서 | 열 순서 |
|---|---|---|---|
| `outputs/folds.csv` | 54,728행 × (`index`, `author`, `fold`) | 원본 `index` 오름차순 (train_clean+val_clean 합친 뒤 정렬) | — |
| `outputs/oof/{실험}.npy` | (54,728, 5) float64 | **folds.csv 와 같은 행 순서** (i번째 행 = folds.csv i번째 행) | 작가 0,1,2,3,4 확률 |
| `outputs/test_pred/{실험}.npy` | (19,617, 5) float64 | `test_clean.csv` = `sample_submission.csv` 행 순서 (index 0→19616) | 작가 0,1,2,3,4 확률 |
| `results/nb_results.csv` | — | 어떤 실험이 몇 점인지 | — |

```python
import numpy as np, pandas as pd
folds = pd.read_csv("outputs/folds.csv")             # 행 i ↔ OOF 행 i
y, fold = folds["author"].to_numpy(), folds["fold"].to_numpy()
oof = np.load("outputs/oof/<실험>.npy")               # (54728, 5)
test = np.load("outputs/test_pred/<실험>.npy")         # (19617, 5)
# 스태킹: 메타 모델도 같은 fold 로 나눠서 학습/검증해야 누수가 없다
```

- 각 행의 확률 합 = 1. NB(보정 없음)는 0 또는 1 에 매우 가까운 값이 많으니 soft voting 시 주의(보정 버전 권장).
- 학습 데이터는 전처리 담당 정리본(중복·라벨충돌 151행 제거, 54,879 → 54,728행) 기준이다.
  데이콘 원본 `train.csv` 행 수(54,879)와 다르니, 다른 모델의 OOF 와 합칠 때 `index` 로 맞출 것.

## 7. 결과 요약 (전체 54,728행, 5-fold OOF)

전체 표는 `results/nb_summary.md`, 원자료는 `results/nb_results.csv` (실험 40개).
실행 시간: 1~4단계 전체 170초 + 보충 실험 38초 (24코어, 폴드 5개 병렬).

**최고 설정 → 제출 파일 `outputs/submissions/nb_best.csv`**

| 실험 | 설정 | OOF logloss | accuracy | macro-F1 |
|---|---|---|---|---|
| `mnb_tfidf_word12_a0.03_caliso` | MultinomialNB + TF-IDF 단어 (1,2)-gram, α=0.03, isotonic 보정 | **0.4725** | 0.8326 | 0.8257 |
| `mnb_tfidf_word12_a0.03` | 위와 같고 보정 없음 | 0.4900 | 0.8291 | 0.8219 |

단계별 핵심

1. **모델·벡터화·단위 (α=1)**: logloss 는 MNB + TF-IDF 단어 (1,2) 가 0.876 으로 가장 좋았다. 반면 Count 기반은
   정확도는 높은데(Count 단어 (1,2) acc 0.805) logloss 가 매우 나빴다(1.30, 글자 n-gram Count 3.5~4.1, BNB 글자 9.0).
   특징이 많을수록(bigram, 글자 n-gram) "독립"이라 가정한 증거를 더 많이 곱해서 확률이 0/1 로 극단화되기 때문
   (NB 의 확률 보정 문제).
2. **alpha (라플라스 스무딩)**: TF-IDF 는 작은 alpha(0.01~0.03)가 좋았다. TF-IDF 값이 0~1 사이라 α=1 은
   스무딩이 지나치기 때문이다. Count 단어 (1,1) 은 α=0.3, BNB 는 α=0.03 이 가장 좋았다.
   최고는 MNB + TF-IDF 단어 (1,2), α=0.03 의 0.490 (그래프: `results/alpha_logloss.png`).
3. **전처리 (불용어·문장부호)**: 제거하면 모두 나빠졌다. 단어 (1,2) 기준 유지 0.490 → 불용어 제거 0.573 →
   문장부호 제거 0.588 → 둘 다 0.691. 글자 n-gram 은 문장부호 제거 손해가 더 컸다(0.662 → 0.800).
   기능어와 문장부호(따옴표 모양 등)가 작가의 문체 신호라는 가설과 맞는 결과다.
4. **확률 보정**: 최고 설정은 isotonic 이 0.490 → 0.4725 로 조금 좋아졌고, sigmoid 는 0.506 으로 오히려 나빠졌다.
   과신이 심한 모델에서는 효과가 컸다(보충 실험): MNB Count 단어 (1,2) 는 1.296 → 0.552 (isotonic),
   BNB Count 단어 (1,1) 은 1.373 → 0.712. 정확도는 거의 그대로이고 확률 크기만 바로잡힌다.

참고·주의

- **긴 문장일수록 쉽다**: 테스트 길이 범위(≥131자) 학습 행만 보면 최고 설정의 OOF logloss 가 0.286 이다.
  테스트 문장이 더 길기 때문에 데이콘 점수는 OOF 0.47 보다 좋게 나올 수 있다(제출 전이라 확인 안 됨).
- **isotonic 보정은 정확히 0 인 확률을 낸다** (테스트 9,825행에 0 이 포함됨). 그 칸이 정답이면 그 행의 logloss 가
  폭증한다. OOF 에서는 8행이 이런 경우였다. 확률 하한을 1e-4 로 두면 OOF logloss 가 0.4725 → 0.4687 로 좋아져서
  `outputs/submissions/nb_best_clip0.0001.csv` (`make_submission.py --clip 1e-4`) 도 만들어 두었다.
  OOF 로 정한 후처리이므로 두 파일을 모두 제출해 비교해 볼 수 있다.
- **팀 검증 방식 차이**: 전처리 담당 README 는 고정 80:20 검증셋(`val_clean`) 으로 logloss 를 비교하라고 되어 있다.
  여기서는 합의된 5-fold(`folds.csv`)를 썼으므로, 같은 모델이어도 숫자가 그 방식과 정확히 같지는 않다.
  앙상블 담당과 비교할 때는 같은 `folds.csv` 기준 OOF 끼리 비교할 것.
- 테스트 환경: Python 3.12, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2.

## 8. 앙상블 단계

NB 단계와 **같은 `outputs/folds.csv`·같은 저장 형식**으로 기본 모델을 더 만들고, 수업에서 배운 앙상블
세 가지(배깅, 투표 기반 분류기, 부스팅)를 적용했다. 나이브 베이즈는 NB 단계 최고 설정(`mnb_tfidf_word12_a0.03_caliso`)의
OOF/테스트 확률을 그대로 재사용하고 다시 학습하지 않았다.

### 8-1. 실행 방법

```bash
python src/run_base.py                   # 4-1 기본 모델: LR(단어·글자) × C, 선형 SVM + 확률 보정 × C
python src/run_bagging.py                # 4-2 배깅: MNB × 10, LR(OOF logloss 최저 설정) × 10
python src/run_voting.py                 # 4-3 soft voting: 저장된 확률 평균 (단순 / 가중)
python src/run_stacking.py               # 4-4 스태킹: LightGBM(부스팅) / 로지스틱 회귀 메타 모델
python src/report_ensemble.py            # 4-5 results/ensemble_summary.md + results/ensemble_compare.png
python src/make_ensemble_submission.py   # 4-6 outputs/submissions/ensemble_best.csv (+ .json)
```

- 설정(C 후보, 글자 n-gram 범위·특징 수 상한, 배깅 모델 수, 폴드 병렬 수)은 `src/config.py` 의 "앙상블 단계 설정" 한 곳에 있다.
- 이미 끝난 실험은 건너뛴다. 다시 돌리려면 `--force`, 실험 하나만: `python src/run_base.py --exp svm_tfidf_word12_C0.1_calsig --force`.
- 기본 모델은 `--n-jobs`(기본 2) 개 폴드를 동시에 돌린다. 글자 n-gram 은 폴드마다 메모리를 많이 써서 노트북(RAM 16GB)에 맞춰 2로 두었다.
- 배깅 모델 수를 줄이려면 `python src/run_bagging.py --n-estimators 5` (결과표 `note` 열에 줄인 사실이 기록됨).
- 빠른 점검: 각 스크립트에 `--sample 2000`. 샘플 모드는 폴드를 새로 만들기 때문에 NB 최고 설정도 먼저
  `python src/run_nb.py --exp mnb_tfidf_word12_a0.03_caliso mnb_tfidf_word12_a0.03 --sample 2000` 으로 만들어야 한다.
- 특정 실험으로 제출 파일: `python src/make_ensemble_submission.py --exp <실험이름>` → `outputs/submissions/<실험이름>.csv`

### 8-2. 각 방식 (수업 개념과 대응)

| 단계 | 수업 개념 | 내용 |
|---|---|---|
| 4-1 기본 모델 | (앙상블 재료) | 로지스틱 회귀(다항, softmax) — TF-IDF 단어 (1,2)-gram / 글자 char_wb (1,5)-gram(max_features 20만), C ∈ {1, 4, 16}. 선형 SVM — `CalibratedClassifierCV(LinearSVC, sigmoid)`, TF-IDF 단어 (1,2)-gram, C ∈ {0.1, 0.5}. NB 는 재사용 |
| 4-2 배깅 | 배깅 | `BaggingClassifier(estimator, n_estimators=10, max_samples=0.8, bootstrap=True)` — 복원추출 표본마다 모델을 따로 학습하고 확률 평균. MNB(α=0.03)와 LR(단어 C=16) 각각, 같은 설정의 단일 모델과 비교 |
| 4-3 soft voting | 투표 기반 분류기 | 저장된 OOF/테스트 확률을 평균 (재학습 없음). 단순 평균 vs 가중 평균. 가중치는 OOF logloss 최소화 (`scipy.optimize.minimize`, SLSQP, 합 1·0 이상) |
| 4-4 스태킹 | 부스팅 | 기본 모델 4개(NB, LR 단어, LR 글자, SVM)의 OOF log 확률을 옆으로 붙인 20열이 입력, 작가가 타깃. 메타 모델 LightGBM(num_leaves 15, learning_rate 0.05, early stopping) / 비교용 로지스틱 회귀. 같은 `folds.csv` 로 다시 5-fold |

- **누수 방지**: 벡터라이저는 각 폴드의 학습 부분에만 fit. SVM 은 벡터라이저까지 Pipeline 에 넣어서 확률 보정용 내부 5-fold 에서도
  내부 학습 부분에만 fit (NB 단계 보정과 같은 방식). LightGBM early stopping 은 검증 폴드가 아니라 학습 부분에서 떼어낸 10% 로 판단.
- **hard voting**: 다수결은 클래스 하나만 고르고 확률을 내지 않으므로 logloss 로 평가할 수 없어 하지 않았다.
- **스태킹 입력을 log 확률로 둔 이유**: 트리(LightGBM)는 단조 변환에 결과가 같고, 로지스틱 회귀 메타 모델에는
  "모델별 확률의 가중 기하평균"을 배우는 꼴이 된다. isotonic 보정 NB 의 0 확률은 1e-7 로 잘라서 log 를 취했다.

### 8-3. 결과 파일

- OOF / 테스트 확률: NB 단계와 같은 `outputs/oof/{실험}.npy` (54,728 × 5), `outputs/test_pred/{실험}.npy` (19,617 × 5)
- `results/ensemble_results.csv` 열: `exp`(실험이름) | `method`(단일/배깅/soft voting/스태킹) | `models`(사용 모델) | `settings`(설정) |
  `logloss` | `accuracy` | `macro_f1` | `time_sec`(소요시간) | `logloss_fold_std` | `logloss_testlike`(긴문장 logloss) | `note`(비고) | …
- `time_sec`: 5개 폴드의 벡터화+학습+예측 시간 합 (노트북 Intel Core Ultra 5 226V, 8스레드, 폴드 2개 동시 실행).
  NB 단계 시간은 다른 PC(24코어)에서 잰 값이라 직접 비교할 수 없다.
  `results/run_ensemble_base.log` 마지막의 총 소요(13,675s)에는 실행 중 노트북이 절전 상태였던 약 3시간 20분이 들어 있다.
  그 사이에 돌던 `svm_tfidf_word12_C0.1_calsig` 는 `run_ensemble_svm_rerun.log` 로 다시 돌려 시간을 쟀다 (logloss 는 같음).

### 8-4. 결과 요약 (전체 54,728행, 같은 5-fold OOF)

전체 표는 `results/ensemble_summary.md`, 원자료는 `results/ensemble_results.csv` (실험 20개). 그래프: `results/ensemble_compare.png`.

| 방식 | 최고 실험 | OOF logloss | accuracy | macro-F1 | 긴문장 logloss |
|---|---|---|---|---|---|
| 나이브 베이즈 (NB 단계 최고) | `mnb_tfidf_word12_a0.03_caliso` | 0.4725 | 0.8326 | 0.8257 | 0.2863 |
| 단일 모델 (LR / SVM) | `lr_tfidf_word12_C16` | 0.4603 | 0.8401 | 0.8326 | 0.2886 |
| 배깅 | `bag10_lr_tfidf_word12_C16` | 0.4894 | 0.8334 | 0.8253 | 0.3286 |
| soft voting | `vote_wopt_nb-lrw-lrc-svm` | 0.4191 | 0.8622 | 0.8562 | 0.2598 |
| **스태킹 (LightGBM 메타)** | `stack_lgbm_nb-lrw-lrc-svm` | **0.3806** | **0.8655** | **0.8601** | **0.2047** |

단계별 핵심

1. **기본 모델**: 로지스틱 회귀 단어 (1,2)-gram 이 C=1 → 4 → 16 에서 0.618 → 0.500 → 0.460 으로 좋아져 NB 최고(0.4725)보다 낮았다.
   C=16 이 후보 중 가장 커서, 더 큰 C 가 더 좋을 수 있다 (확인 안 함). 글자 n-gram LR 은 0.634 → 0.544 → 0.540 (특징 134,962개,
   상한 20만에 안 걸림), SVM + sigmoid 보정은 C=0.1 0.542, C=0.5 0.476.
2. **배깅**: 두 모델 모두 단일보다 나빠졌다. MNB 0.4900 → 0.5082 (+0.018), LR 0.4603 → 0.4894 (+0.029), 정확도도 조금 떨어짐.
   분석: 복원추출 80% 표본은 서로 다른 행이 약 55% (1−e^−0.8) 뿐이라 모델마다 데이터가 줄어든다. NB·선형 모델은 원래 데이터가 조금
   바뀌어도 예측이 크게 흔들리지 않는(분산이 작은) 모델이라 평균으로 줄일 분산이 적고, 데이터가 줄어든 손해가 더 컸다.
   배깅은 깊은 결정트리처럼 분산이 큰 모델에서 효과가 크다.
3. **soft voting**: 단순 평균은 NB + LR 단어 0.4207 이 가장 좋았고, 모델을 더 넣으면 오히려 나빠졌다(4개 0.4283) — 약한 모델에도
   같은 비중을 주기 때문. 가중 평균은 4개 조합이 0.4191 로 가장 좋았고 가중치는 NB 0.455 · LR 단어 0.446 · LR 글자 0.099 · SVM 0.000.
   폴드마다 나머지 폴드로 가중치를 다시 찾아 검증한 값도 0.4192 라서 가중치 과적합은 거의 없다.
4. **스태킹**: LightGBM 메타 0.3806 (폴드별 best_iteration 119~169), 로지스틱 회귀 메타 0.3850. 가중 평균(0.4191)보다 크게 좋다.
   분석: 메타 모델은 모델별 비중뿐 아니라 확률 크기(보정)와 작가별로 어느 모델을 믿을지까지 배운다.

주의

- **스태킹 OOF 는 약간 낙관적일 수 있다**: 메타 모델을 1단계 OOF 를 만든 것과 같은 폴드로 검증하는데, 메타 모델의 학습 행에 붙은
  1단계 OOF 확률은 검증 폴드 행까지 포함해 학습한 기본 모델이 만든 것이라 검증 폴드 정보가 조금 섞인다. 가중 voting 의 가중치도
  OOF 전체로 찾았다(다만 위처럼 폴드별 재검증 값과 거의 같다).
- **테스트는 문장이 길어 모든 모델이 더 확신한다**: 평균 최대확률이 OOF → 테스트에서 NB 0.828 → 0.892, LR 단어 0.823 → 0.866,
  스태킹(LightGBM) 0.870 → 0.926. 메타 모델은 OOF(짧은 문장 위주) 분포로 학습했으므로 데이콘 점수가 OOF 순서와 같을지는 제출해 봐야 안다.
  긴문장 logloss(테스트 최소 길이 이상 학습 행)에서도 스태킹이 0.2047 로 가장 좋았다.

### 8-5. 최종 제출 파일

- `outputs/submissions/ensemble_best.csv` = `stack_lgbm_nb-lrw-lrc-svm` 의 테스트 확률 (OOF logloss 최저, 어떤 실험인지는 `ensemble_best.json`)
  - 19,617행 × (`index`, `0`~`4`), 열 순서 = `sample_submission.csv`, 각 행 합 1, 확률 최솟값 7.83e-05
- 비교용 NB 단일 최고 제출 파일 `outputs/submissions/nb_best.csv` 는 그대로 둠
- 앙상블 단계 환경: Python 3.13.2, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, lightgbm 4.7.0
