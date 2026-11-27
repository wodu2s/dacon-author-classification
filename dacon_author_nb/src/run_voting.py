"""4-3. 투표 기반 분류기 (soft voting): 저장된 OOF/테스트 확률을 평균해 최종 확률을 정한다 (모델 재학습 없음).

  python src/run_voting.py

  - 단순 평균: 모든 모델에 같은 가중치
  - 가중 평균: OOF logloss 가 최소가 되도록 가중치를 찾음 (scipy.optimize.minimize, SLSQP, 가중치 합 1·0 이상)
    가중치를 OOF 전체로 찾으므로 그 OOF logloss 는 약간 낙관적이다. 그래서 "폴드마다 나머지 4개 폴드로
    가중치를 다시 찾아 남은 폴드에 적용"한 logloss 도 note 열에 함께 기록한다.
  - hard voting(다수결)은 클래스 하나만 내고 확률이 없어 logloss 로 평가할 수 없으므로 하지 않는다.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from config import CLASSES, N_FOLDS, add_common_args
from ensemble_common import (BASE_LABEL, METHOD_VOTE, combo_label, load_ens_results, load_pred, pick_bases,
                             record, setup)

COMBOS = [("nb", "lrw"), ("nb", "lrc"), ("nb", "lrw", "svm"), ("nb", "lrw", "lrc", "svm")]
EPS = 1e-15


def fit_weights(p_true: np.ndarray) -> np.ndarray:
    """p_true: (행, 모델) 각 모델이 정답 작가에 준 확률.
    가중 평균 확률의 logloss = -mean(log(p_true @ w)) 를 w ≥ 0, sum(w) = 1 조건에서 최소화."""
    m = p_true.shape[1]

    def f(w):
        return -np.mean(np.log(np.clip(p_true @ w, EPS, None)))

    def grad(w):
        return -(p_true / np.clip(p_true @ w, EPS, None)[:, None]).mean(0)

    r = minimize(f, np.full(m, 1 / m), jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * m,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(m)}],
                 options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    args = ap.parse_args()
    ctx = setup(args.data, args.sample)
    bases = pick_bases(ctx)
    print("[재료] " + ", ".join(f"{c}={n}" for c, n in bases.items()))
    preds = {c: load_pred(ctx, n) for c, n in bases.items()}
    rows = np.arange(len(ctx.y))

    for combo in COMBOS:
        key = "-".join(combo)
        oofs = [preds[c][0] for c in combo]
        tests = [preds[c][1] for c in combo]
        print(f"\n[{key}]")

        # soft voting (단순 평균): 모델별 확률을 똑같은 비중으로 평균해 최종 확률 결정
        t0 = time.perf_counter()
        oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
        record(ctx, f"vote_mean_{key}", METHOD_VOTE, combo_label(combo, bases), "단순 평균 (가중치 동일)",
               oof, test, time.perf_counter() - t0)

        # soft voting (가중 평균): 잘 맞히는 모델에 더 큰 비중 → 가중치는 OOF logloss 최소화로 결정
        t0 = time.perf_counter()
        p_true = np.stack([o[rows, ctx.y] for o in oofs], axis=1)   # 각 모델이 정답에 준 확률
        w = fit_weights(p_true)
        oof = np.tensordot(w, np.stack(oofs), axes=1)
        test = np.tensordot(w, np.stack(tests), axes=1)
        # 정직한 점검: 폴드 k 의 가중치는 나머지 폴드 OOF 로만 찾아서 폴드 k 에 적용
        nested = np.zeros_like(oof)
        for k in range(N_FOLDS):
            va = ctx.fold_ids == k
            wk = fit_weights(p_true[~va])
            nested[va] = np.tensordot(wk, np.stack([o[va] for o in oofs]), axes=1)
        nested_ll = log_loss(ctx.y, nested, labels=CLASSES)
        w_txt = ", ".join(f"{BASE_LABEL[c]} {wi:.3f}" for c, wi in zip(combo, w))
        record(ctx, f"vote_wopt_{key}", METHOD_VOTE, combo_label(combo, bases),
               f"가중 평균 w=[{w_txt}] (OOF logloss 최소화, SLSQP, 합=1·0 이상)", oof, test,
               time.perf_counter() - t0,
               note=f"가중치를 OOF 전체로 찾음(약간 낙관적). 폴드별로 가중치를 다시 찾아 검증한 logloss {nested_ll:.4f}")

    res = load_ens_results(ctx.paths)
    vote = res[res["method"] == METHOD_VOTE].sort_values("logloss")
    print("\n===== soft voting 결과 (OOF logloss 순) =====")
    print(vote[["exp", "logloss", "accuracy", "macro_f1", "settings"]].to_string(index=False))


if __name__ == "__main__":
    main()
