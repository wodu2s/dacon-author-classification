# 소설 작가 분류 — 나이브 베이즈 파트

응용머신러닝(2026-2) 기말 팀 프로젝트 · 월간 데이콘 「소설 작가 분류 AI 경진대회」(235670)
영어 소설 문장 → 작가 0~4 다중분류, 평가지표 **multi-class logloss**.

팀 진행: ① 텍스트 전처리 → **② 나이브 베이즈 (이 폴더)** → ③ 앙상블

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
│   └── make_submission.py   # 최고 설정 → 제출 csv
├── outputs/
│   ├── folds.csv            # ★ 팀 공유: 행 index → fold 번호
│   ├── oof/{실험이름}.npy     # ★ 학습 전체 OOF 확률 (54,728 × 5)
│   ├── test_pred/{실험이름}.npy # ★ 테스트 확률 = 5개 폴드 모델 평균 (19,617 × 5)
│   └── submissions/
│       ├── nb_best.csv                 # ★ 데이콘 제출용 (+ nb_best.json: 어떤 실험인지)
│       └── nb_best_clip0.0001.csv      # 확률 하한 1e-4 적용본 (7절 참고)
├── results/
│   ├── nb_results.csv       # 실험 결과표 (실험마다 한 줄)
│   ├── nb_summary.md        # 단계별 비교표 (보고서용)
│   ├── alpha_logloss.png    # alpha 에 따른 logloss 그래프 (보고서용)
│   └── run_full.log, run_extra_calib.log   # 실행 로그
└── requirements.txt
```

## 2. 실행 방법

```bash
pip install -r requirements.txt          # pandas, numpy, scipy, scikit-learn, matplotlib

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
