"""실험 정의(이름 <-> 설정), 벡터라이저/모델 생성, 단계별 실험 목록.

실험 이름 규칙:  {모델}_{벡터라이저}_{단위}{n-gram 하한}{상한}_a{alpha}[_p{전처리}][_cal{보정}]
  예) mnb_count_word12_a0.1          MultinomialNB + CountVectorizer + 단어 (1,2)-gram, alpha=0.1
      bnb_count_charwb24_a0.3        BernoulliNB + 글자(char_wb) (2,4)-gram
      mnb_tfidf_word12_a0.03_pnostop 불용어 제거 버전
      mnb_count_charwb15_a0.1_calsig 시그모이드(Platt) 확률 보정 버전
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline

from config import (ALPHAS, CALIB_INNER_FOLDS, CALIB_METHODS, CLASSES, DEFAULT_ALPHA, MIN_DF, PREPS, SEED,
                    STAGE1_UNITS, STAGE2_TOP_K, TFIDF_SUBLINEAR, WORD_TOKEN_PATTERN)

MODEL_LABEL = {"mnb": "MultinomialNB", "bnb": "BernoulliNB"}
VEC_LABEL = {"count": "CountVectorizer", "tfidf": "TfidfVectorizer"}
ANALYZER_LABEL = {"word": "word", "charwb": "char_wb", "char": "char"}
CALIB_CODE = {"none": "", "sigmoid": "sig", "isotonic": "iso"}

_NAME_RE = re.compile(
    r"^(?P<model>mnb|bnb)_(?P<vec>count|tfidf)_(?P<an>word|charwb|char)(?P<n1>\d)(?P<n2>\d)"
    r"_a(?P<alpha>\d+(?:\.\d+)?(?:e-?\d+)?)"
    r"(?:_p(?P<prep>nostopnopunct|nostop|nopunct|keep))?"
    r"(?:_cal(?P<cal>sig|iso))?$"
)


@dataclass(frozen=True)
class Exp:
    model: str              # mnb | bnb
    vec: str                # count | tfidf
    analyzer: str           # word | charwb | char
    ngram: tuple            # (하한, 상한)
    alpha: float            # 라플라스 스무딩 강도
    prep: str = "keep"      # 전처리 변형 (data.apply_prep)
    calib: str = "none"     # none | sigmoid | isotonic

    @property
    def name(self) -> str:
        s = f"{self.model}_{self.vec}_{self.analyzer}{self.ngram[0]}{self.ngram[1]}_a{self.alpha:g}"
        if self.prep != "keep":
            s += f"_p{self.prep}"
        if self.calib != "none":
            s += f"_cal{CALIB_CODE[self.calib]}"
        return s

    @property
    def feature_key(self) -> tuple:
        """같은 특징 행렬을 쓰는 실험끼리 묶는 키 (alpha·모델만 다르면 벡터화를 한 번만 한다)."""
        return (self.vec, self.analyzer, self.ngram, self.prep)

    @classmethod
    def from_name(cls, name: str) -> "Exp":
        m = _NAME_RE.match(name)
        if not m:
            raise ValueError(f"실험 이름 형식이 잘못됨: {name}\n  예) mnb_count_word12_a0.1, "
                             "bnb_count_charwb24_a0.3_pnostop, mnb_count_charwb15_a0.1_calsig")
        cal = {None: "none", "sig": "sigmoid", "iso": "isotonic"}[m["cal"]]
        return cls(m["model"], m["vec"], m["an"], (int(m["n1"]), int(m["n2"])), float(m["alpha"]),
                   m["prep"] or "keep", cal)


# ---------------------------------------------------------------------------
# 벡터화 (텍스트 분석: BoW / TF-IDF)
# ---------------------------------------------------------------------------
def build_vectorizer(e: Exp):
    kw = dict(lowercase=True, ngram_range=e.ngram, min_df=MIN_DF)
    if e.analyzer == "word":
        kw.update(analyzer="word", token_pattern=WORD_TOKEN_PATTERN)   # 단어 n-gram (+문장부호 토큰)
    elif e.analyzer == "charwb":
        kw.update(analyzer="char_wb")   # 글자 n-gram (단어 경계 안에서만, 철자·어미·문장부호 습관 포착)
    else:
        kw.update(analyzer="char")      # 글자 n-gram (단어 경계를 넘어서도 생성)
    if e.vec == "count":
        return CountVectorizer(**kw)    # BoW: 문서별 n-gram 등장 횟수
    # TF-IDF: 많은 문서에 흔한 n-gram 의 가중치를 낮추고 드문 n-gram 을 강조
    return TfidfVectorizer(sublinear_tf=TFIDF_SUBLINEAR, **kw)


# ---------------------------------------------------------------------------
# 나이브 베이즈
#   - MultinomialNB : P(단어|작가) 를 "빈도"로 추정 → 문서 = 단어 빈도 벡터
#   - BernoulliNB   : P(단어 출현|작가) 를 "있다/없다"로 추정 → 안 나온 단어도 (1-p) 로 확률에 반영
#     (binarize=0 이므로 Count/TF-IDF 를 넣어도 똑같이 0/1 로 바뀜 → BNB×TF-IDF 는 중복이라 단계 실험에서 생략)
#   - alpha (라플라스 스무딩): 학습에 안 나온 n-gram 의 확률이 0 이 되어 곱 전체가 0 이 되는 문제를 막는다.
#     alpha 가 클수록 0 확률 문제 완화·확률이 평평해짐, 작을수록 관측 빈도를 그대로 믿음(과적합·극단 확률).
# ---------------------------------------------------------------------------
def build_model(e: Exp):
    if e.model == "mnb":
        return MultinomialNB(alpha=e.alpha)
    return BernoulliNB(alpha=e.alpha, binarize=0.0)


def build_calibrated(e: Exp):
    """확률 보정: NB 는 '특징 독립' 가정 때문에 같은 증거를 중복으로 곱해 확률이 0/1 로 극단화된다
    → logloss 에서 크게 손해. 내부 CV 로 만든 (누수 없는) NB 예측 확률을 입력으로
    sigmoid(Platt) 또는 isotonic 회귀를 학습해 확률 크기를 다시 맞춘다.

    벡터라이저까지 Pipeline 에 넣어서, 보정기 학습용 내부 폴드에서도 벡터라이저를 내부 학습 부분에만 fit 한다.
    ensemble=False: 내부 5-fold 의 out-of-fold 예측으로 보정기를 학습하고, NB 본체는 폴드 학습 데이터 전체로 1번 학습.
    """
    pipe = Pipeline([("vec", build_vectorizer(e)), ("nb", build_model(e))])
    inner_cv = StratifiedKFold(n_splits=CALIB_INNER_FOLDS, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(estimator=pipe, method=e.calib, cv=inner_cv, ensemble=False)


# ---------------------------------------------------------------------------
# 단계별 실험 목록 (조합 폭발을 막기 위해 단계적으로 좁힘)
# ---------------------------------------------------------------------------
def stage1_exps() -> list[Exp]:
    """1단계: 모델 × 벡터라이저 × 단위 를 기본 alpha 로 비교."""
    exps = []
    for analyzer, ngram in STAGE1_UNITS:
        exps.append(Exp("mnb", "count", analyzer, ngram, DEFAULT_ALPHA))
        exps.append(Exp("mnb", "tfidf", analyzer, ngram, DEFAULT_ALPHA))
        exps.append(Exp("bnb", "count", analyzer, ngram, DEFAULT_ALPHA))
    return exps


def _plain(res: pd.DataFrame) -> pd.DataFrame:
    return res[(res["calibration"] == "none")]


def stage2_exps(res: pd.DataFrame) -> list[Exp]:
    """2단계: 1단계 상위 K개 + (모델,벡터라이저) 조합별 1위 설정에서 alpha 탐색.
    (기본 alpha=1 은 TF-IDF 값(0~1 사이)에 비해 스무딩이 과해 TF-IDF 가 불리하므로 조합별 1위도 함께 탐색)"""
    names1 = {e.name for e in stage1_exps()}
    base = res[res["exp"].isin(names1)]
    if base.empty:
        raise SystemExit("2단계에는 1단계 결과가 필요합니다. 먼저 --stage 1 을 실행하세요.")
    top = base.nsmallest(STAGE2_TOP_K, "logloss")["exp"].tolist()
    fam = base.loc[base.groupby(["model", "vectorizer"])["logloss"].idxmin(), "exp"].tolist()
    chosen = list(dict.fromkeys(top + fam))
    return [replace(Exp.from_name(n), alpha=a) for n in chosen for a in ALPHAS]


def stage3_exps(res: pd.DataFrame) -> list[Exp]:
    """3단계: 최고 설정(+ 다른 단위(단어↔글자)의 최고 설정)에서 전처리 변형 비교."""
    base = _plain(res)
    base = base[base["prep"] == "keep"]
    if base.empty:
        raise SystemExit("3단계에는 1·2단계 결과가 필요합니다.")
    best = Exp.from_name(base.nsmallest(1, "logloss")["exp"].iloc[0])
    is_word = base["analyzer"] == "word"
    other = base[~is_word] if best.analyzer == "word" else base[is_word]
    chosen = [best] + ([Exp.from_name(other.nsmallest(1, "logloss")["exp"].iloc[0])] if len(other) else [])
    return [replace(e, prep=p) for e in chosen for p in PREPS]


def stage4_exps(res: pd.DataFrame) -> list[Exp]:
    """4단계: 보정 전 logloss 최고 설정에 sigmoid / isotonic 확률 보정.
    + 정확도 최고 설정이 다르면 그것도 보정 (NB 는 분류(순위)는 잘해도 확률이 극단적이라 logloss 가 나쁜 경우가
      많음 → 보정으로 확률 크기만 고치면 logloss 최고가 될 수 있는지 확인)."""
    base = _plain(res)
    if base.empty:
        raise SystemExit("4단계에는 앞 단계 결과가 필요합니다.")
    best = Exp.from_name(base.nsmallest(1, "logloss")["exp"].iloc[0])
    best_acc = Exp.from_name(base.nlargest(1, "accuracy")["exp"].iloc[0])
    chosen = list(dict.fromkeys([best, best_acc]))
    return [replace(e, calib=m) for e in chosen for m in CALIB_METHODS]


STAGES = {1: stage1_exps, 2: stage2_exps, 3: stage3_exps, 4: stage4_exps}


def exp_row_fields(e: Exp) -> dict:
    return dict(exp=e.name, model=MODEL_LABEL[e.model], vectorizer=VEC_LABEL[e.vec],
                analyzer=ANALYZER_LABEL[e.analyzer], ngram=f"({e.ngram[0]},{e.ngram[1]})",
                alpha=e.alpha, calibration=e.calib, prep=e.prep)


def check_classes(classes) -> None:
    # predict_proba 열 순서 = model.classes_ 순서 → 반드시 0,1,2,3,4 여야 제출/앙상블 열 순서가 맞음
    assert np.array_equal(np.asarray(classes), np.asarray(CLASSES)), f"classes_ 순서 이상: {classes}"
