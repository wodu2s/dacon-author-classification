"""4-4. 부스팅(스태킹 방식): 기본 모델들의 OOF 확률을 입력으로, 작가를 맞히는 메타 모델을 학습한다.

  python src/run_stacking.py

  - 입력: 기본 모델(NB, LR 단어, LR 글자, SVM)의 OOF 확률을 옆으로 붙인 행렬 (모델 4개 × 5열 = 20열)
          확률 대신 log 확률을 넣는다. 트리(LightGBM)는 단조 변환에 결과가 같고, 로지스틱 회귀에는
          "모델별 확률의 가중 기하평균"을 배우는 꼴이 되어 더 자연스럽다.
  - 메타 모델 ① LightGBM (그래디언트 부스팅): num_leaves 15, learning_rate 0.05, early stopping
            ② 로지스틱 회귀 (단순 스태킹, 비교용)
  - 검증: 같은 outputs/folds.csv 로 다시 5-fold → 메타 모델의 OOF / 테스트 확률(5개 폴드 평균) 생성
  - early stopping 은 검증 폴드가 아니라 학습 부분에서 떼어낸 10% 로 판단한다 (검증 폴드로 멈추면 누수).
  - 주의: 1단계 OOF 를 만든 것과 같은 폴드로 메타 모델을 검증하므로, 결과가 약간 낙관적일 수 있다.
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from config import CLASSES, N_CLASSES, N_FOLDS, SEED, add_common_args
from ensemble_common import METHOD_STACK, combo_label, load_ens_results, load_pred, pick_bases, record, setup

COMBO = ("nb", "lrw", "lrc", "svm")
LOG_CLIP = 1e-7          # log(0) 방지 (isotonic 보정 NB 는 정확히 0 인 확률을 낸다)
LGBM_PARAMS = dict(objective="multiclass", num_leaves=15, learning_rate=0.05,
                   n_estimators=3000, min_child_samples=20, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
EARLY_STOP = 100         # 검증 logloss 가 100 라운드 동안 좋아지지 않으면 멈춤
LR_META_C = 1.0


def fit_meta(kind: str, X_tr, y_tr):
    if kind == "lgbm":
        # 부스팅: 앞 트리들이 틀린 부분(잔차/그래디언트)을 다음 트리가 순차적으로 보완
        es_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        fit_idx, es_idx = next(es_split.split(X_tr, y_tr))
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr[fit_idx], y_tr[fit_idx], eval_X=X_tr[es_idx], eval_y=y_tr[es_idx], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return model, int(model.best_iteration_)
    # 단순 스태킹: 기본 모델 log 확률의 선형 결합 → softmax
    return LogisticRegression(C=LR_META_C, max_iter=1000).fit(X_tr, y_tr), None


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--meta", nargs="+", choices=["lgbm", "lr"], default=["lgbm", "lr"])
    args = ap.parse_args()
    ctx = setup(args.data, args.sample)
    bases = pick_bases(ctx)
    preds = [load_pred(ctx, bases[c]) for c in COMBO]
    # 스태킹 입력: 기본 모델 OOF 확률(log)을 옆으로 붙임 → (학습 행, 모델 수 × 5)
    X = np.hstack([np.log(np.clip(o, LOG_CLIP, 1)) for o, _ in preds])
    X_test = np.hstack([np.log(np.clip(t, LOG_CLIP, 1)) for _, t in preds])
    print(f"[스태킹 입력] {combo_label(COMBO, bases)}  →  학습 {X.shape}, 테스트 {X_test.shape}")

    key = "-".join(COMBO)
    for kind in args.meta:
        print(f"\n[메타 모델: {kind}]")
        t0 = time.perf_counter()
        oof = np.full((len(ctx.y), N_CLASSES), np.nan)
        test = np.zeros((len(ctx.test), N_CLASSES))
        iters = []
        for k, (tr, va) in enumerate(ctx.splits):   # 1단계와 같은 folds.csv 로 메타 모델도 5-fold
            model, it = fit_meta(kind, X[tr], ctx.y[tr])
            oof[va] = model.predict_proba(X[va])
            test += model.predict_proba(X_test) / N_FOLDS
            iters.append(it)
            print(f"   fold {k} 완료 | logloss {log_loss(ctx.y[va], oof[va], labels=CLASSES):.4f}"
                  + (f" | best_iteration {it}" if it else ""), flush=True)
        if kind == "lgbm":
            settings = (f"LightGBM 메타 (num_leaves=15, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, "
                        f"early stopping {EARLY_STOP}라운드 · 학습 부분의 10%로 판단), 입력=기본 모델 log 확률")
            note = f"폴드별 best_iteration {iters}; 1단계와 같은 폴드로 검증해 약간 낙관적일 수 있음"
            models = "LightGBM ← " + combo_label(COMBO, bases)
        else:
            settings = f"LogisticRegression 메타 (C={LR_META_C:g}), 입력=기본 모델 log 확률"
            note = "1단계와 같은 폴드로 검증해 약간 낙관적일 수 있음"
            models = "LogisticRegression ← " + combo_label(COMBO, bases)
        record(ctx, f"stack_{kind}_{key}", METHOD_STACK, models, settings, oof, test,
               time.perf_counter() - t0, note=note)

    res = load_ens_results(ctx.paths)
    st = res[res["method"] == METHOD_STACK].sort_values("logloss")
    print("\n===== 스태킹 결과 =====")
    print(st[["exp", "logloss", "accuracy", "macro_f1", "time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
