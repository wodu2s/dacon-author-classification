"""최고 설정(OOF logloss 최저)의 테스트 확률로 데이콘 제출 파일 생성.

  python src/make_submission.py                         # → outputs/submissions/nb_best.csv
  python src/make_submission.py --exp mnb_count_word12_a0.1   # 특정 실험 → outputs/submissions/<실험이름>.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import CLASSES, ID_COL, add_common_args, get_paths
from data import load_data


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--exp", help="제출할 실험 이름 (생략하면 OOF logloss 최저 실험)")
    ap.add_argument("--clip", type=float, default=None,
                    help="확률 하한 (예: 1e-4). isotonic 보정은 정확히 0인 확률을 내는데, 그 칸이 정답이면 "
                         "logloss 가 폭증하므로 하한을 두고 다시 합=1 로 맞춘다. 파일명에 _clip 이 붙음")
    args = ap.parse_args()
    paths = get_paths(args.data, args.sample)
    res = pd.read_csv(paths.results_csv)
    name = args.exp or res.nsmallest(1, "logloss")["exp"].iloc[0]
    row = res.set_index("exp").loc[name]

    _, test, sub = load_data(args.data)
    pred = np.load(paths.test_dir / f"{name}.npy")
    assert pred.shape == (len(sub), 5), f"테스트 예측 shape 이상: {pred.shape}"
    if args.clip:
        y = pd.read_csv(paths.folds)["author"].to_numpy()
        oof = np.load(paths.oof_dir / f"{name}.npy")
        oof_c = np.clip(oof, args.clip, 1)
        oof_c /= oof_c.sum(axis=1, keepdims=True)
        print(f"  clip {args.clip:g}: OOF logloss {log_loss(y, oof, labels=CLASSES):.4f} → "
              f"{log_loss(y, oof_c, labels=CLASSES):.4f}")
        pred = np.clip(pred, args.clip, 1)
    pred = pred / pred.sum(axis=1, keepdims=True)  # 각 행 합 = 1

    out = sub.copy()
    for col in [c for c in sub.columns if c != ID_COL]:
        out[col] = pred[:, int(col)]               # 열 이름 "k" ← 클래스 k 확률 (sample_submission 열 순서 유지)
    assert (out[ID_COL].to_numpy() == test[ID_COL].to_numpy()).all()

    fname = ("nb_best" if not args.exp else name) + (f"_clip{args.clip:g}" if args.clip else "")
    csv_path = paths.sub_dir / f"{fname}.csv"
    out.to_csv(csv_path, index=False)
    info = dict(exp=name, clip=args.clip, oof_logloss=float(row["logloss"]), oof_accuracy=float(row["accuracy"]),
                oof_macro_f1=float(row["macro_f1"]), test_pred=f"outputs/test_pred/{name}.npy")
    if not args.exp:
        (paths.sub_dir / f"{fname}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    s = out.drop(columns=ID_COL).sum(axis=1)
    print(f"제출 파일 → {csv_path}")
    print(f"  실험: {name}  (OOF logloss {row['logloss']:.4f}, acc {row['accuracy']:.4f})")
    print(f"  shape {out.shape}, 열 {list(out.columns)}, 행 합 min {s.min():.6f} / max {s.max():.6f}")
    print(f"  예측 작가 분포: {np.bincount(pred.argmax(1), minlength=5).tolist()}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
