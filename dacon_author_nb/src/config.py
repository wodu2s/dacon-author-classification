"""프로젝트 설정 (경로·열 이름·시드·폴드 수·실험 축) — 설정은 이 파일 한 곳에서만 바꾼다."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# 데이터 버전: 파일 경로만 바꾸면 원본 <-> 전처리본을 교체할 수 있도록 한 곳에서 관리
#   - clean : 전처리 담당이 준 정리본 (공백 정리 + 중복/라벨충돌 행 제거, 문장부호·불용어는 유지)
#             train_clean + val_clean 을 합쳐 전체 학습 데이터로 보고 5-fold 를 나눈다.
#   - raw   : 데이콘 원본을 data/raw/ 에 직접 넣었을 때
# ---------------------------------------------------------------------------
DATASETS = {
    "clean": {
        "train_files": [DATA_PROCESSED / "train_clean.csv", DATA_PROCESSED / "val_clean.csv"],
        "test_file": DATA_PROCESSED / "test_clean.csv",
        "sample_submission": DATA_PROCESSED / "sample_submission.csv",
    },
    "raw": {
        "train_files": [DATA_RAW / "train.csv"],
        "test_file": DATA_RAW / "test_x.csv",
        "sample_submission": DATA_RAW / "sample_submission.csv",
    },
}
DATA_VERSION = "clean"  # 기본 데이터 버전 (--data 로 덮어쓸 수 있음)

ID_COL = "index"     # 행 식별자 열
TEXT_COL = "text"    # 텍스트 열
LABEL_COL = "author"  # 라벨 열 (작가 0~4)

N_CLASSES = 5
CLASSES = list(range(N_CLASSES))
SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# 벡터화(BoW/TF-IDF) 공통 설정
# ---------------------------------------------------------------------------
MIN_DF = 2               # 2개 미만 문서에만 나온 n-gram 은 버림 (희귀 잡음 제거, 전처리 담당과 동일)
TFIDF_SUBLINEAR = True   # TF-IDF 에서 tf -> 1 + log(tf) (전처리 담당 설정과 동일)
# 단어 n-gram 토큰 규칙: 단어(1글자 포함) + 문장부호 1개씩을 각각 토큰으로 본다.
# sklearn 기본값(\b\w\w+\b)은 문장부호와 "I", "a" 를 버리는데, 작가 분류에서는 이것도 문체 신호다.
WORD_TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"

# ---------------------------------------------------------------------------
# 나이브 베이즈 실험 축 (단계적으로 좁혀 간다)
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 1.0                      # 1단계 비교용 alpha (sklearn 기본값 = 고전적 라플라스 add-one)
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]     # 2단계 alpha(라플라스 스무딩) 탐색 범위
STAGE1_UNITS = [("word", (1, 1)), ("word", (1, 2)), ("charwb", (2, 4)), ("charwb", (1, 5))]
STAGE2_TOP_K = 3                         # 2단계: 1단계 상위 K개 + (모델,벡터라이저) 조합별 1위
PREPS = ["keep", "nostop", "nopunct", "nostopnopunct"]  # 3단계: 불용어/문장부호 제거 O/X
CALIB_METHODS = ["sigmoid", "isotonic"]  # 4단계: 확률 보정 방법
CALIB_INNER_FOLDS = 5                    # 보정기 학습용 내부 CV 폴드 수

N_JOBS = 5  # 폴드를 병렬로 돌릴 프로세스 수

# ---------------------------------------------------------------------------
# 앙상블 단계 설정 (③ 앙상블: 기본 모델 → 배깅 → soft voting → 스태킹)
# ---------------------------------------------------------------------------
NB_BEST = "mnb_tfidf_word12_a0.03_caliso"   # NB 단계 최고 설정 (OOF/테스트 확률을 재사용, 다시 학습하지 않음)
NB_PLAIN = "mnb_tfidf_word12_a0.03"         # 같은 설정의 보정 없는 단일 NB (배깅 비교 기준)
NB_BEST_ALPHA = 0.03                        # 배깅에 넣을 MultinomialNB 의 alpha (NB 단계 최고값)
LR_CS = [1.0, 4.0, 16.0]                    # 로지스틱 회귀 규제 강도 C 후보 (클수록 규제 약함)
SVM_CS = [0.1, 0.5]                         # 선형 SVM C 후보
LR_MAX_ITER = 1000
CHAR_NGRAM = (1, 5)                         # 글자(char_wb) n-gram 범위
CHAR_MAX_FEATURES = 200_000                 # 글자 n-gram 특징 수 상한 (실행 시간·메모리 관리)
BAG_N_ESTIMATORS = 10                       # 배깅: 부트스트랩 표본으로 학습할 모델 수
BAG_MAX_SAMPLES = 0.8                       # 배깅: 모델 하나가 보는 표본 비율
ENS_N_JOBS = 2                              # 기본 모델 폴드 병렬 수 (노트북 메모리 고려, --n-jobs 로 변경)


def get_paths(data_version: str = DATA_VERSION, sample: int | None = None) -> SimpleNamespace:
    """출력 경로 모음. 데이터 버전/샘플 모드마다 폴더를 분리해 결과가 섞이지 않게 한다."""
    root = ROOT if data_version == "clean" else ROOT / f"run_{data_version}"
    if sample:
        root = root / f"sample_run_{sample}"
    p = SimpleNamespace(
        root=root,
        folds=root / "outputs" / "folds.csv",
        oof_dir=root / "outputs" / "oof",
        test_dir=root / "outputs" / "test_pred",
        sub_dir=root / "outputs" / "submissions",
        results_dir=root / "results",
        results_csv=root / "results" / "nb_results.csv",
        alpha_plot=root / "results" / "alpha_logloss.png",
        summary_md=root / "results" / "nb_summary.md",
        ens_results_csv=root / "results" / "ensemble_results.csv",   # 앙상블 단계 결과표 (누적)
        ens_summary_md=root / "results" / "ensemble_summary.md",
        ens_plot=root / "results" / "ensemble_compare.png",
    )
    for d in (p.oof_dir, p.test_dir, p.sub_dir, p.results_dir):
        d.mkdir(parents=True, exist_ok=True)
    return p


def add_common_args(parser):
    parser.add_argument("--data", default=DATA_VERSION, choices=list(DATASETS),
                        help="데이터 버전 (기본: %(default)s)")
    parser.add_argument("--sample", type=int, default=None,
                        help="학습 데이터를 N행만 층화 샘플링해 빠르게 파이프라인 점검 (결과는 sample_run_N/ 에 따로 저장)")
    return parser
