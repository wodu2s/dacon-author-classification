"""앙상블 단계 공통 모듈: 데이터·폴드 준비, 기본 모델(LR / SVM / 배깅) 생성, 5-fold 실행, 결과 저장.

NB 단계와 같은 규칙을 따른다.
  - 폴드: outputs/folds.csv 를 그대로 사용 (모델끼리 OOF 를 합치려면 같은 폴드여야 함)
  - 누수 방지: 벡터라이저는 각 폴드의 학습 부분에만 fit, 검증 폴드·테스트는 transform 만
  - 저장: outputs/oof/{실험}.npy (학습 행 × 5), outputs/test_pred/{실험}.npy (테스트 행 × 5, 5개 폴드 평균)
  - 결과표: results/ensemble_results.csv 에 실험마다 한 줄씩 누적
"""
from __future__ import annotations

import re
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import (BAG_MAX_SAMPLES, CALIB_INNER_FOLDS, CHAR_MAX_FEATURES, CLASSES, LABEL_COL, LR_MAX_ITER, MIN_DF,
                    N_CLASSES, N_FOLDS, NB_BEST, NB_PLAIN, SEED, TEXT_COL, TFIDF_SUBLINEAR, WORD_TOKEN_PATTERN,
                    get_paths)
from cv import compute_metrics
from data import load_data, load_or_make_folds
from experiments import check_classes

ENS_COLUMNS = ["exp", "method", "models", "settings", "logloss", "accuracy", "macro_f1", "time_sec",
               "logloss_fold_std", "logloss_testlike", "note", "data_version", "n_train", "finished_at"]

METHOD_SINGLE, METHOD_BAG, METHOD_VOTE, METHOD_STACK = "단일", "배깅", "soft voting", "스태킹"


# ---------------------------------------------------------------------------
# 데이터·폴드 준비
# ---------------------------------------------------------------------------
def setup(data_version: str, sample: int | None) -> SimpleNamespace:
    """학습/테스트 데이터와 팀 공통 폴드(outputs/folds.csv)를 읽어 실행에 필요한 것을 한 번에 묶는다."""
    paths = get_paths(data_version, sample)
    train, test, sub = load_data(data_version, sample)
    folds = load_or_make_folds(train, paths.folds)   # 이미 있으면 읽고 행(index/author)이 같은지 검사
    y = train[LABEL_COL].to_numpy()
    fold_ids = folds["fold"].to_numpy()
    splits = [(np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k)) for k in range(N_FOLDS)]
    long_mask = train[TEXT_COL].str.len().to_numpy() >= test[TEXT_COL].str.len().min()
    print(f"[데이터] 학습 {len(train):,}행 / 테스트 {len(test):,}행  |  폴드 {paths.folds}  |  출력 {paths.root}")
    return SimpleNamespace(paths=paths, train=train, test=test, sub=sub, y=y, fold_ids=fold_ids, splits=splits,
                           long_mask=long_mask, data_version=data_version, sample=sample,
                           texts=train[TEXT_COL].to_numpy(dtype=object),
                           test_texts=test[TEXT_COL].to_numpy(dtype=object))


# ---------------------------------------------------------------------------
# 결과 저장 (results/ensemble_results.csv)
# ---------------------------------------------------------------------------
def load_ens_results(paths) -> pd.DataFrame:
    if paths.ens_results_csv.exists():
        return pd.read_csv(paths.ens_results_csv)
    return pd.DataFrame(columns=ENS_COLUMNS)


def is_done(name: str, ctx, res: pd.DataFrame) -> bool:
    return (name in set(res["exp"]) and (ctx.paths.oof_dir / f"{name}.npy").exists()
            and (ctx.paths.test_dir / f"{name}.npy").exists())


def record(ctx, name: str, method: str, models: str, settings: str, oof: np.ndarray, test_pred: np.ndarray,
           time_sec: float, note: str = "") -> dict:
    """OOF·테스트 확률을 NB 단계와 같은 형식으로 저장하고, 지표를 계산해 결과표에 한 줄 추가한다."""
    assert oof.shape == (len(ctx.y), N_CLASSES), f"OOF shape 이상: {oof.shape}"
    assert test_pred.shape == (len(ctx.test), N_CLASSES), f"테스트 shape 이상: {test_pred.shape}"
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any(), "확률에 NaN 이 있음"
    assert np.allclose(oof.sum(1), 1, atol=1e-6) and np.allclose(test_pred.sum(1), 1, atol=1e-6), "행 합이 1 이 아님"
    np.save(ctx.paths.oof_dir / f"{name}.npy", oof)
    np.save(ctx.paths.test_dir / f"{name}.npy", test_pred)
    m = compute_metrics(ctx.y, oof, ctx.fold_ids, ctx.long_mask)
    row = dict(exp=name, method=method, models=models, settings=settings, logloss=m["logloss"],
               accuracy=m["accuracy"], macro_f1=m["macro_f1"], time_sec=round(time_sec, 1),
               logloss_fold_std=m["logloss_fold_std"], logloss_testlike=m["logloss_testlike"], note=note,
               data_version=ctx.data_version, n_train=len(ctx.y),
               finished_at=datetime.now().isoformat(timespec="seconds"))
    res = load_ens_results(ctx.paths)
    res = res[res["exp"] != name]
    new = pd.DataFrame([row], columns=ENS_COLUMNS)
    res = new if res.empty else pd.concat([res, new], ignore_index=True)
    res.to_csv(ctx.paths.ens_results_csv, index=False)
    print(f"  ✔ {name:<40} logloss {m['logloss']:.4f} (±{m['logloss_fold_std']:.4f})  acc {m['accuracy']:.4f}  "
          f"F1 {m['macro_f1']:.4f}  긴문장 {m['logloss_testlike']:.4f}  ({time_sec:.0f}s)"
          + (f"  [{note}]" if note else ""), flush=True)
    return row


def load_pred(ctx, name: str):
    """저장된 OOF·테스트 확률 읽기 (재학습 없이 앙상블에 재사용)."""
    po, pt = ctx.paths.oof_dir / f"{name}.npy", ctx.paths.test_dir / f"{name}.npy"
    if not (po.exists() and pt.exists()):
        hint = (f"python src/run_nb.py --exp {NB_BEST} {NB_PLAIN}" if name in (NB_BEST, NB_PLAIN)
                else "python src/run_base.py")
        if ctx.sample:
            hint += f" --sample {ctx.sample}"
        raise SystemExit(f"'{name}' 의 OOF/테스트 확률이 없습니다 ({po}).\n→ 먼저 실행: {hint}")
    oof, test_pred = np.load(po), np.load(pt)
    assert oof.shape == (len(ctx.y), N_CLASSES) and test_pred.shape == (len(ctx.test), N_CLASSES)
    return oof, test_pred


BASE_LABEL = {"nb": "NB", "lrw": "LR 단어", "lrc": "LR 글자", "svm": "SVM"}


def pick_bases(ctx) -> dict:
    """앙상블 재료: NB 단계 최고 설정 + 기본 모델 계열별(LR 단어 / LR 글자 / SVM) OOF logloss 최저 설정."""
    res = load_ens_results(ctx.paths)
    single = res[res["method"] == METHOD_SINGLE]
    out = {"nb": NB_BEST}
    for code, pat in [("lrw", r"^lr_tfidf_word"), ("lrc", r"^lr_tfidf_charwb"), ("svm", r"^svm_")]:
        sub = single[single["exp"].str.match(pat)]
        if sub.empty:
            raise SystemExit(f"'{code}' 계열 결과가 없습니다 → 먼저 python src/run_base.py 실행")
        out[code] = sub.nsmallest(1, "logloss")["exp"].iloc[0]
    return out


def combo_label(combo, bases) -> str:
    return " + ".join(f"{c}={bases[c]}" for c in combo)


def nb_logloss(ctx, name: str) -> float | None:
    """NB 단계 결과표(nb_results.csv)에서 실험의 OOF logloss 를 읽는다."""
    if not ctx.paths.results_csv.exists():
        return None
    nb = pd.read_csv(ctx.paths.results_csv).set_index("exp")
    return float(nb.loc[name, "logloss"]) if name in nb.index else None


# ---------------------------------------------------------------------------
# 기본 모델 정의 (이름 <-> 설정)
#   lr_tfidf_word12_C4                 로지스틱 회귀 + TF-IDF 단어 (1,2)-gram, C=4
#   lr_tfidf_charwb15_C4               로지스틱 회귀 + TF-IDF 글자(char_wb) (1,5)-gram
#   svm_tfidf_word12_C0.1_calsig       선형 SVM + sigmoid 확률 보정
#   bag10_mnb_tfidf_word12_a0.03       MultinomialNB 배깅 (모델 10개)
#   bag10_lr_tfidf_word12_C4           로지스틱 회귀 배깅 (모델 10개)
# ---------------------------------------------------------------------------
_SPEC_RE = re.compile(r"^(?:bag(?P<n>\d+)_)?(?P<model>lr|svm|mnb)_tfidf_(?P<an>word|charwb)(?P<n1>\d)(?P<n2>\d)"
                      r"_(?:C(?P<C>[\d.e-]+)|a(?P<alpha>[\d.e-]+))(?:_calsig)?$")


@dataclass(frozen=True)
class BaseSpec:
    model: str                 # lr | svm | mnb
    analyzer: str              # word | charwb
    ngram: tuple               # (하한, 상한)
    C: float | None = None     # LR / SVM 규제 강도
    alpha: float | None = None  # MNB 라플라스 스무딩
    n_estimators: int = 0      # 0 = 단일 모델, >0 = 배깅 모델 수

    @property
    def unit(self) -> str:
        return f"{self.analyzer}{self.ngram[0]}{self.ngram[1]}"

    @property
    def name(self) -> str:
        core = f"{self.model}_tfidf_{self.unit}_" + (f"a{self.alpha:g}" if self.model == "mnb" else f"C{self.C:g}")
        if self.model == "svm":
            core += "_calsig"
        return f"bag{self.n_estimators}_{core}" if self.n_estimators else core

    @property
    def base_name(self) -> str:
        """배깅 모델이면 비교 대상인 단일 모델 이름."""
        return BaseSpec(self.model, self.analyzer, self.ngram, self.C, self.alpha).name

    @property
    def feature_key(self) -> tuple:
        return (self.analyzer, self.ngram)

    @property
    def method(self) -> str:
        return METHOD_BAG if self.n_estimators else METHOD_SINGLE

    @property
    def models_label(self) -> str:
        base = {"lr": "LogisticRegression", "svm": "LinearSVC + CalibratedClassifierCV(sigmoid)",
                "mnb": "MultinomialNB"}[self.model]
        return f"BaggingClassifier({base}) × {self.n_estimators}" if self.n_estimators else base

    @property
    def settings(self) -> str:
        unit = "단어" if self.analyzer == "word" else "글자 char_wb"
        s = f"TF-IDF {unit} {self.ngram}, min_df={MIN_DF}"
        if self.analyzer != "word":
            s += f", max_features={CHAR_MAX_FEATURES:,}"
        s += f", alpha={self.alpha:g}" if self.model == "mnb" else f", C={self.C:g}"
        if self.model == "svm":
            s += f", 보정 내부 {CALIB_INNER_FOLDS}-fold"
        if self.n_estimators:
            s += f", n_estimators={self.n_estimators}, max_samples={BAG_MAX_SAMPLES}, bootstrap=True"
        return s

    @classmethod
    def from_name(cls, name: str) -> "BaseSpec":
        m = _SPEC_RE.match(name)
        if not m:
            raise ValueError(f"실험 이름 형식이 잘못됨: {name}  (예: lr_tfidf_word12_C4, bag10_lr_tfidf_word12_C4)")
        return cls(m["model"], m["an"], (int(m["n1"]), int(m["n2"])),
                   float(m["C"]) if m["C"] else None, float(m["alpha"]) if m["alpha"] else None,
                   int(m["n"]) if m["n"] else 0)


def build_tfidf(analyzer: str, ngram: tuple) -> TfidfVectorizer:
    """TF-IDF 벡터라이저 (NB 단계와 같은 설정: 소문자화, min_df=2, sublinear_tf)."""
    kw = dict(lowercase=True, ngram_range=ngram, min_df=MIN_DF, sublinear_tf=TFIDF_SUBLINEAR)
    if analyzer == "word":
        kw.update(analyzer="word", token_pattern=WORD_TOKEN_PATTERN)   # 단어 n-gram (+문장부호 토큰)
    else:
        kw.update(analyzer="char_wb", max_features=CHAR_MAX_FEATURES)  # 글자 n-gram, 특징 수 상한
    return TfidfVectorizer(**kw)


def build_model(spec: BaseSpec, inner_jobs: int | None = None):
    """이미 벡터화된 행렬에 학습할 모델 (SVM 은 build_svm 사용)."""
    if spec.model == "lr":
        # 로지스틱 회귀(다항, softmax): 선형 결정경계 + 확률을 직접 출력 → predict_proba 사용 가능
        base = LogisticRegression(C=spec.C, max_iter=LR_MAX_ITER)
    elif spec.model == "mnb":
        base = MultinomialNB(alpha=spec.alpha)
    else:
        raise ValueError(spec.model)
    if not spec.n_estimators:
        return base
    # 배깅(Bootstrap AGGregatING): 학습 데이터에서 복원추출한 표본(80%)마다 같은 모델을 따로 학습하고
    # 예측 확률을 평균한다 → 모델마다 다른 데이터를 봐서 생기는 흔들림(분산)이 평균으로 줄어든다.
    return BaggingClassifier(estimator=base, n_estimators=spec.n_estimators, max_samples=BAG_MAX_SAMPLES,
                             bootstrap=True, n_jobs=inner_jobs, random_state=SEED)


def build_svm(spec: BaseSpec) -> CalibratedClassifierCV:
    """선형 SVM + 확률 보정. LinearSVC 는 확률(predict_proba)이 없고 결정 함수 값만 낸다
    → CalibratedClassifierCV(sigmoid, Platt scaling)로 결정 함수 값을 확률로 바꾼다.
    벡터라이저까지 Pipeline 에 넣어서, 보정용 내부 폴드에서도 벡터라이저를 내부 학습 부분에만 fit (NB 단계와 동일).
    ensemble=False: 내부 5-fold 의 out-of-fold 결정값으로 보정기를 학습하고, SVM 본체는 폴드 학습 데이터 전체로 1번 학습."""
    pipe = Pipeline([("vec", build_tfidf(spec.analyzer, spec.ngram)), ("svm", LinearSVC(C=spec.C))])
    inner_cv = StratifiedKFold(n_splits=CALIB_INNER_FOLDS, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=inner_cv, ensemble=False)


# ---------------------------------------------------------------------------
# 폴드 1개 처리 (병렬 프로세스에서 실행)
# ---------------------------------------------------------------------------
def _fit_counting_warnings(model, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    return sum(issubclass(x.category, ConvergenceWarning) for x in w)


def _fold_vectorized(k, tr_idx, va_idx, texts, y, test_texts, specs, inner_jobs):
    """같은 특징(단위·n-gram)을 쓰는 모델 여러 개(C 만 다름 등)를 한 번의 벡터화로 처리."""
    t0 = time.perf_counter()
    vec = build_tfidf(*specs[0].feature_key)
    X_tr = vec.fit_transform(texts[tr_idx])   # 데이터 누수 방지: 어휘·IDF 는 학습 폴드로만 fit
    X_va = vec.transform(texts[va_idx])       # 검증 폴드는 transform 만
    X_te = vec.transform(test_texts)          # 테스트도 transform 만
    vec_time = time.perf_counter() - t0
    out = {}
    for s in specs:
        t1 = time.perf_counter()
        model = build_model(s, inner_jobs)
        n_warn = _fit_counting_warnings(model, X_tr, y[tr_idx])
        check_classes(model.classes_)
        out[s.name] = (model.predict_proba(X_va), model.predict_proba(X_te), time.perf_counter() - t1, n_warn)
    return dict(k=k, va_idx=va_idx, vec_time=vec_time, n_features=X_tr.shape[1], out=out)


def _fold_svm(k, tr_idx, va_idx, texts, y, test_texts, specs, inner_jobs):
    (s,) = specs
    t0 = time.perf_counter()
    est = build_svm(s)
    n_warn = _fit_counting_warnings(est, texts[tr_idx], y[tr_idx])
    check_classes(est.classes_)
    p_va, p_te = est.predict_proba(texts[va_idx]), est.predict_proba(test_texts)
    n_feat = len(est.calibrated_classifiers_[0].estimator.named_steps["vec"].vocabulary_)
    return dict(k=k, va_idx=va_idx, vec_time=0.0, n_features=n_feat,
                out={s.name: (p_va, p_te, time.perf_counter() - t0, n_warn)})


# ---------------------------------------------------------------------------
# 실행: 5-fold → OOF(검증 폴드 행 자리에 예측) + 테스트(5개 폴드 모델 평균)
# ---------------------------------------------------------------------------
def run_specs(specs: list[BaseSpec], ctx, n_jobs: int, inner_jobs: int | None = None, force: bool = False,
              extra_note: dict | None = None) -> None:
    res = load_ens_results(ctx.paths)
    specs = list(dict.fromkeys(specs))
    todo = [s for s in specs if force or not is_done(s.name, ctx, res)]
    if len(specs) - len(todo):
        print(f"이미 끝난 실험 {len(specs) - len(todo)}개는 건너뜀 (--force 로 재실행)")
    jobs = OrderedDict()
    for s in todo:
        key = ("svm", s.name) if s.model == "svm" else ("vec",) + s.feature_key + (s.n_estimators > 0,)
        jobs.setdefault(key, []).append(s)

    t_all = time.perf_counter()
    for j, (key, group) in enumerate(jobs.items(), 1):
        worker = _fold_svm if key[0] == "svm" else _fold_vectorized
        print(f"\n[{j}/{len(jobs)}] {', '.join(s.name for s in group)}", flush=True)
        t_job = time.perf_counter()
        acc = {s.name: dict(oof=np.full((len(ctx.y), N_CLASSES), np.nan), test=np.zeros((len(ctx.test), N_CLASSES)),
                            model_time=0.0, warn=0) for s in group}
        vec_time, n_feats = 0.0, []
        gen = Parallel(n_jobs=min(n_jobs, N_FOLDS), return_as="generator_unordered")(
            delayed(worker)(k, tr, va, ctx.texts, ctx.y, ctx.test_texts, group, inner_jobs)
            for k, (tr, va) in enumerate(ctx.splits))
        for r in gen:
            vec_time += r["vec_time"]
            n_feats.append(r["n_features"])
            msg = []
            for name, (p_va, p_te, t_model, n_warn) in r["out"].items():
                acc[name]["oof"][r["va_idx"]] = p_va      # OOF: 검증 폴드 행 자리에 예측 확률을 채움
                acc[name]["test"] += p_te / N_FOLDS         # 테스트: 5개 폴드 모델 예측의 평균
                acc[name]["model_time"] += t_model
                acc[name]["warn"] += n_warn
                msg.append(f"{name}={log_loss(ctx.y[r['va_idx']], p_va, labels=CLASSES):.4f}")
            print(f"   fold {r['k']} 완료 ({time.perf_counter() - t_job:.0f}s) | 특징 {r['n_features']:,}개 | "
                  f"logloss " + ", ".join(msg), flush=True)
        for s in group:
            a = acc[s.name]
            notes = [f"특징 {int(np.mean(n_feats)):,}개(폴드 평균)"]
            if a["warn"]:
                notes.append(f"수렴 경고 {a['warn']}회(max_iter={LR_MAX_ITER})")
            if extra_note and s.name in extra_note:
                notes.append(extra_note[s.name])
            record(ctx, s.name, s.method, s.models_label, s.settings, a["oof"], a["test"],
                   vec_time + a["model_time"], note="; ".join(notes))
        print(f"  (이 묶음 {time.perf_counter() - t_job:.0f}s, 누적 {time.perf_counter() - t_all:.0f}s)", flush=True)


def add_n_jobs_args(parser, default: int, help_text: str):
    parser.add_argument("--n-jobs", type=int, default=default, help=help_text + " (기본 %(default)s)")
    parser.add_argument("--force", action="store_true", help="이미 끝난 실험도 다시 실행")
    return parser
