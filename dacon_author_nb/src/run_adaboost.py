"""수업 범위 4-1. 부스팅(AdaBoost): 나이브 베이즈를 약한 학습기로 순차 학습.

  python src/run_adaboost.py                  # n_estimators {50,100} × learning_rate {0.5,1.0} = 4개
  python src/run_adaboost.py --sample 2000    # 2,000행으로 빠른 흐름 점검

  - 모델: AdaBoostClassifier(estimator=MultinomialNB(alpha=0.03)), 입력은 TF-IDF 단어 (1,2)-gram (NB 단계와 같은 벡터화)
  - 폴드: outputs/folds.csv, 벡터라이저는 각 폴드의 학습 부분에만 fit (누수 방지)
  - 저장: outputs/oof/{실험}.npy, outputs/test_pred/{실험}.npy, results/ensemble_results.csv 에 한 줄씩 추가
  - 비교 기준: 보정 없는 단일 NB (config.NB_PLAIN)

확률에 대한 주의 (sklearn 1.9 의 AdaBoost 는 SAMME 방식만 있음):
  predict_proba 는 각 NB 의 확률이 아니라 "예측 작가(hard label)의 가중 투표" f(x) 를 (클래스 수-1)=4 로 나눠 softmax 한다.
  NB 가 모두 같은 작가를 골라도 그 작가 확률이 최대 0.255 라서, 정확도와 상관없이 OOF logloss 는 1.368 보다 작아질 수 없다.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB

from config import (ADA_LEARNING_RATES, ADA_N_ESTIMATORS, CLASSES, ENS_N_JOBS, MIN_DF, N_CLASSES, N_FOLDS,
                    NB_BEST_ALPHA, NB_PLAIN, SEED, add_common_args)
from ensemble_common import (METHOD_BOOST, add_n_jobs_args, build_tfidf, is_done, load_ens_results, nb_logloss,
                             record, setup)
from experiments import check_classes


def ada_name(n: int, lr: float) -> str:
    return f"ada{n}_lr{lr:g}_mnb_tfidf_word12_a{NB_BEST_ALPHA:g}"


def _fold(k, tr_idx, va_idx, texts, y, test_texts, configs):
    """폴드 1개: 벡터화는 한 번만 하고, AdaBoost 설정 여러 개를 같은 특징 행렬로 학습."""
    t0 = time.perf_counter()
    vec = build_tfidf("word", (1, 2))
    X_tr = vec.fit_transform(texts[tr_idx])   # 데이터 누수 방지: 어휘·IDF 는 학습 폴드로만 fit
    X_va = vec.transform(texts[va_idx])       # 검증 폴드·테스트는 transform 만
    X_te = vec.transform(test_texts)
    vec_time = time.perf_counter() - t0
    out = {}
    for n, lr in configs:
        t1 = time.perf_counter()
        # 부스팅(AdaBoost, SAMME): NB 를 하나씩 차례로 학습한다. 앞 NB 가 틀린 문장의 가중치를 키워서
        # 다음 NB 는 그 문장들에 더 집중하게 하고(sample_weight), 최종 예측은 정확한 NB 일수록 큰 가중치로 투표한다.
        model = AdaBoostClassifier(estimator=MultinomialNB(alpha=NB_BEST_ALPHA), n_estimators=n,
                                   learning_rate=lr, random_state=SEED)
        model.fit(X_tr, y[tr_idx])
        check_classes(model.classes_)
        out[ada_name(n, lr)] = (model.predict_proba(X_va), model.predict_proba(X_te), time.perf_counter() - t1,
                                len(model.estimators_))
    return dict(k=k, va_idx=va_idx, vec_time=vec_time, n_features=X_tr.shape[1], out=out)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--n-estimators", nargs="+", type=int, default=ADA_N_ESTIMATORS)
    ap.add_argument("--learning-rate", nargs="+", type=float, default=ADA_LEARNING_RATES)
    # AdaBoost 는 순차 학습이라 n_jobs 가 없다 → 폴드 단위로 병렬 (폴드마다 메모리를 써서 노트북에선 2)
    add_n_jobs_args(ap, ENS_N_JOBS, "폴드 병렬 프로세스 수")
    args = ap.parse_args()

    ctx = setup(args.data, args.sample)
    res = load_ens_results(ctx.paths)
    configs = [(n, lr) for n in args.n_estimators for lr in args.learning_rate]
    todo = [c for c in configs if args.force or not is_done(ada_name(*c), ctx, res)]
    if len(configs) - len(todo):
        print(f"이미 끝난 실험 {len(configs) - len(todo)}개는 건너뜀 (--force 로 재실행)")

    t0 = time.perf_counter()
    if todo:
        print(f"\n[AdaBoost] {', '.join(ada_name(*c) for c in todo)}", flush=True)
        acc = {ada_name(*c): dict(oof=np.full((len(ctx.y), N_CLASSES), np.nan),
                                  test=np.zeros((len(ctx.test), N_CLASSES)), model_time=0.0, n_used=[])
               for c in todo}
        vec_time, n_feats = 0.0, []
        gen = Parallel(n_jobs=min(args.n_jobs, N_FOLDS), return_as="generator_unordered")(
            delayed(_fold)(k, tr, va, ctx.texts, ctx.y, ctx.test_texts, todo) for k, (tr, va) in enumerate(ctx.splits))
        for r in gen:
            vec_time += r["vec_time"]
            n_feats.append(r["n_features"])
            msg = []
            for name, (p_va, p_te, t_model, n_used) in r["out"].items():
                acc[name]["oof"][r["va_idx"]] = p_va      # OOF: 검증 폴드 행 자리에 예측 확률
                acc[name]["test"] += p_te / N_FOLDS         # 테스트: 5개 폴드 모델 평균
                acc[name]["model_time"] += t_model
                acc[name]["n_used"].append(n_used)
                msg.append(f"{name}={log_loss(ctx.y[r['va_idx']], p_va, labels=CLASSES):.4f}"
                           f"/acc {np.mean(p_va.argmax(1) == ctx.y[r['va_idx']]):.4f}")
            print(f"   fold {r['k']} 완료 ({time.perf_counter() - t0:.0f}s) | " + ", ".join(msg), flush=True)
        for n, lr in todo:
            name = ada_name(n, lr)
            a = acc[name]
            notes = [f"특징 {int(np.mean(n_feats)):,}개(폴드 평균)",
                     "확률 = SAMME 가중 투표 softmax(f/4) → 최대 0.255, logloss 하한 1.368"]
            if min(a["n_used"]) < n:
                notes.append(f"조기 종료: 폴드별 학습기 수 {a['n_used']}")
            settings = (f"TF-IDF 단어 (1, 2), min_df={MIN_DF}, MultinomialNB alpha={NB_BEST_ALPHA:g}, "
                        f"n_estimators={n}, learning_rate={lr:g}, algorithm=SAMME")
            record(ctx, name, METHOD_BOOST, f"AdaBoostClassifier(MultinomialNB) × {n}", settings,
                   a["oof"], a["test"], vec_time + a["model_time"], note="; ".join(notes))

    # 단일 NB(보정 없음) 대비 비교 — 나빠져도 그대로 보고
    single = nb_logloss(ctx, NB_PLAIN)
    res = load_ens_results(ctx.paths).set_index("exp")
    print(f"\n총 소요 {time.perf_counter() - t0:.0f}s")
    print(f"===== AdaBoost vs 단일 NB {NB_PLAIN} (OOF logloss {single if single is None else f'{single:.4f}'}) =====")
    nb = None
    if ctx.paths.results_csv.exists():
        nbr = pd.read_csv(ctx.paths.results_csv).set_index("exp")
        nb = nbr.loc[NB_PLAIN] if NB_PLAIN in nbr.index else None
    for n, lr in configs:
        r = res.loc[ada_name(n, lr)]
        d = "" if single is None else f"  (차이 {r['logloss'] - single:+.4f})"
        a = "" if nb is None else f"  | 정확도 단일 {nb['accuracy']:.4f} → {r['accuracy']:.4f}"
        print(f"  {ada_name(n, lr):<34} logloss {r['logloss']:.4f}{d}{a}")


if __name__ == "__main__":
    main()
