"""K-겹 교차검증 실행기: OOF 확률 / 테스트 확률(폴드 평균) 생성, 지표 계산, 결과 즉시 저장."""
from __future__ import annotations

import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, log_loss

from config import CLASSES, LABEL_COL, N_CLASSES, N_FOLDS, TEXT_COL
from data import apply_prep
from experiments import Exp, build_calibrated, build_model, build_vectorizer, check_classes, exp_row_fields

RESULT_COLUMNS = ["exp", "model", "vectorizer", "analyzer", "ngram", "alpha", "calibration", "prep",
                  "data_version", "logloss", "logloss_fold_std", "logloss_testlike", "accuracy", "macro_f1",
                  "n_features", "n_train", "time_sec", "finished_at"]


# ---------------------------------------------------------------------------
# 폴드 1개 처리 (병렬 프로세스에서 실행)
# ---------------------------------------------------------------------------
def _fold_plain(k, tr_idx, va_idx, texts, y, test_texts, exps):
    """같은 특징을 쓰는 실험 여러 개(alpha/모델만 다름)를 한 번의 벡터화로 처리."""
    t0 = time.perf_counter()
    vec = build_vectorizer(exps[0])
    X_tr = vec.fit_transform(texts[tr_idx])   # 데이터 누수 방지: 어휘·IDF 는 학습 폴드로만 fit
    X_va = vec.transform(texts[va_idx])       # 검증 폴드는 transform 만
    X_te = vec.transform(test_texts)          # 테스트도 transform 만
    vec_time = time.perf_counter() - t0
    out = {}
    for e in exps:
        t1 = time.perf_counter()
        model = build_model(e).fit(X_tr, y[tr_idx])
        check_classes(model.classes_)
        out[e.name] = (model.predict_proba(X_va), model.predict_proba(X_te), time.perf_counter() - t1)
    return dict(k=k, va_idx=va_idx, vec_time=vec_time, n_features=X_tr.shape[1], out=out)


def _fold_calibrated(k, tr_idx, va_idx, texts, y, test_texts, exps):
    (e,) = exps
    t0 = time.perf_counter()
    est = build_calibrated(e).fit(texts[tr_idx], y[tr_idx])
    check_classes(est.classes_)
    p_va, p_te = est.predict_proba(texts[va_idx]), est.predict_proba(test_texts)
    n_feat = len(est.calibrated_classifiers_[0].estimator.named_steps["vec"].vocabulary_)
    return dict(k=k, va_idx=va_idx, vec_time=0.0, n_features=n_feat,
                out={e.name: (p_va, p_te, time.perf_counter() - t0)})


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------
def load_results(path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def append_result(path, row: dict) -> None:
    res = load_results(path)
    res = res[res["exp"] != row["exp"]]
    new = pd.DataFrame([row], columns=RESULT_COLUMNS)
    res = new if res.empty else pd.concat([res, new], ignore_index=True)
    res.to_csv(path, index=False)


def is_done(e: Exp, paths, res: pd.DataFrame) -> bool:
    return (e.name in set(res["exp"]) and (paths.oof_dir / f"{e.name}.npy").exists()
            and (paths.test_dir / f"{e.name}.npy").exists())


def compute_metrics(y, proba, fold_ids, long_mask) -> dict:
    pred = proba.argmax(1)
    fold_ll = [log_loss(y[fold_ids == k], proba[fold_ids == k], labels=CLASSES) for k in range(N_FOLDS)]
    return dict(
        logloss=log_loss(y, proba, labels=CLASSES),              # 주 지표 (대회 평가지표)
        logloss_fold_std=float(np.std(fold_ll)),
        # 테스트 문장이 학습보다 길다 → 테스트 최소 길이 이상인 학습 행만의 logloss (테스트와 비슷한 조건 참고용)
        logloss_testlike=log_loss(y[long_mask], proba[long_mask], labels=CLASSES),
        accuracy=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro"),
    )


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run_experiments(exps: list[Exp], train: pd.DataFrame, test: pd.DataFrame, folds: pd.DataFrame,
                    paths, data_version: str, n_jobs: int = 5, force: bool = False) -> list[str]:
    y = train[LABEL_COL].to_numpy()
    fold_ids = folds["fold"].to_numpy()
    splits = [(np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k)) for k in range(N_FOLDS)]
    long_mask = train[TEXT_COL].str.len().to_numpy() >= test[TEXT_COL].str.len().min()

    res = load_results(paths.results_csv)
    exps = list(dict.fromkeys(exps))
    todo = [e for e in exps if force or not is_done(e, paths, res)]
    skipped = len(exps) - len(todo)
    if skipped:
        print(f"이미 끝난 실험 {skipped}개는 건너뜀 (--force 로 재실행)")
    if not todo:
        return [e.name for e in exps]

    # 같은 특징(벡터라이저·단위·전처리)을 쓰는 보정 없는 실험끼리 묶어서 벡터화를 한 번만 한다
    jobs = OrderedDict()
    for e in todo:
        key = ("cal", e.name) if e.calib != "none" else ("plain",) + e.feature_key
        jobs.setdefault(key, []).append(e)

    prep_cache = {}

    def texts_for(prep):
        if prep not in prep_cache:
            prep_cache[prep] = (apply_prep(train[TEXT_COL], prep).to_numpy(dtype=object),
                                apply_prep(test[TEXT_COL], prep).to_numpy(dtype=object))
        return prep_cache[prep]

    t_all = time.perf_counter()
    for j, (key, group) in enumerate(jobs.items(), 1):
        worker = _fold_calibrated if key[0] == "cal" else _fold_plain
        tr_texts, te_texts = texts_for(group[0].prep)
        print(f"\n[{j}/{len(jobs)}] {', '.join(e.name for e in group)}", flush=True)
        t_job = time.perf_counter()
        acc = {e.name: dict(oof=np.full((len(y), N_CLASSES), np.nan), test=np.zeros((len(test), N_CLASSES)),
                            model_time=0.0) for e in group}
        vec_time, n_feats = 0.0, []
        gen = Parallel(n_jobs=min(n_jobs, N_FOLDS), return_as="generator_unordered")(
            delayed(worker)(k, tr, va, tr_texts, y, te_texts, group) for k, (tr, va) in enumerate(splits))
        for r in gen:
            vec_time += r["vec_time"]
            n_feats.append(r["n_features"])
            msg = []
            for name, (p_va, p_te, t_model) in r["out"].items():
                acc[name]["oof"][r["va_idx"]] = p_va          # OOF: 검증 폴드 행 자리에 예측 확률을 채움
                acc[name]["test"] += p_te / N_FOLDS             # 테스트: 5개 폴드 모델 예측의 평균
                acc[name]["model_time"] += t_model
                msg.append(f"{name}={log_loss(y[r['va_idx']], p_va, labels=CLASSES):.4f}")
            print(f"   fold {r['k']} 완료 | 특징 {r['n_features']:,}개 | logloss " + ", ".join(msg), flush=True)

        for e in group:
            oof, test_pred = acc[e.name]["oof"], acc[e.name]["test"]
            assert not np.isnan(oof).any(), "OOF 에 비어있는 행이 있음"
            np.save(paths.oof_dir / f"{e.name}.npy", oof)
            np.save(paths.test_dir / f"{e.name}.npy", test_pred)
            m = compute_metrics(y, oof, fold_ids, long_mask)
            row = dict(**exp_row_fields(e), data_version=data_version, **m,
                       n_features=int(np.mean(n_feats)), n_train=len(y),
                       time_sec=round(vec_time + acc[e.name]["model_time"], 1),
                       finished_at=datetime.now().isoformat(timespec="seconds"))
            append_result(paths.results_csv, row)
            print(f"  ✔ {e.name:<34} logloss {m['logloss']:.4f} (±{m['logloss_fold_std']:.4f})  "
                  f"acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}  긴문장 logloss {m['logloss_testlike']:.4f}",
                  flush=True)
        print(f"  (이 묶음 {time.perf_counter() - t_job:.0f}s, 누적 {time.perf_counter() - t_all:.0f}s)", flush=True)
    return [e.name for e in exps]
