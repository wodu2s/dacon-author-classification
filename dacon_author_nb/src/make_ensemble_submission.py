"""4-6. 앙상블 제출 파일: results/ensemble_results.csv 에서 OOF logloss 최저 방법의 테스트 확률로 제출 csv 생성.

  python src/make_ensemble_submission.py                 # → outputs/submissions/ensemble_best.csv (+ .json)
  python src/make_ensemble_submission.py --exp <실험이름>  # 특정 실험 → outputs/submissions/<실험이름>.csv

NB 단일 최고 제출 파일(outputs/submissions/nb_best.csv)은 비교용으로 그대로 둔다.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import ID_COL, N_CLASSES, add_common_args, get_paths
from data import load_data


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--exp", help="제출할 실험 이름 (생략하면 OOF logloss 최저 실험)")
    ap.add_argument("--clip", type=float, default=None, help="확률 하한 (예: 1e-4). 하한 적용 후 다시 합=1 로 맞춤")
    args = ap.parse_args()
    paths = get_paths(args.data, args.sample)
    res = pd.read_csv(paths.ens_results_csv)
    name = args.exp or res.nsmallest(1, "logloss")["exp"].iloc[0]
    row = res.set_index("exp").loc[name]

    _, test, sub = load_data(args.data)
    pred = np.load(paths.test_dir / f"{name}.npy")
    assert pred.shape == (len(sub), N_CLASSES), f"테스트 예측 shape 이상: {pred.shape}"
    if args.clip:
        pred = np.clip(pred, args.clip, 1)
    pred = pred / pred.sum(axis=1, keepdims=True)    # 각 행 합 = 1

    out = sub.copy()
    for col in [c for c in sub.columns if c != ID_COL]:
        out[col] = pred[:, int(col)]                 # 열 이름 "k" ← 작가 k 확률 (sample_submission 열 순서 유지)
    assert (out[ID_COL].to_numpy() == test[ID_COL].to_numpy()).all(), "행 순서가 테스트와 다름"
    assert list(out.columns) == list(sub.columns), "열 순서가 sample_submission 과 다름"
    s = out.drop(columns=ID_COL).sum(axis=1)
    assert np.allclose(s, 1.0), "행 합이 1 이 아님"

    fname = ("ensemble_best" if not args.exp else name) + (f"_clip{args.clip:g}" if args.clip else "")
    csv_path = paths.sub_dir / f"{fname}.csv"
    out.to_csv(csv_path, index=False)
    if not args.exp:
        info = dict(exp=name, method=row["method"], models=row["models"], settings=row["settings"], clip=args.clip,
                    oof_logloss=float(row["logloss"]), oof_accuracy=float(row["accuracy"]),
                    oof_macro_f1=float(row["macro_f1"]), test_pred=f"outputs/test_pred/{name}.npy")
        (paths.sub_dir / f"{fname}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"제출 파일 → {csv_path}")
    print(f"  실험: {name} [{row['method']}]  (OOF logloss {row['logloss']:.4f}, acc {row['accuracy']:.4f})")
    print(f"  shape {out.shape}, 열 {list(out.columns)}, 행 합 min {s.min():.6f} / max {s.max():.6f}, "
          f"확률 최솟값 {pred.min():.2e}")
    print(f"  예측 작가 분포: {np.bincount(pred.argmax(1), minlength=N_CLASSES).tolist()}")
    nb_best = paths.sub_dir / "nb_best.csv"
    print(f"  비교용 NB 제출 파일: {nb_best} ({'있음' if nb_best.exists() else '없음'})")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
