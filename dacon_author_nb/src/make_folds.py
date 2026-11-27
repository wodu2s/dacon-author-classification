"""데이터 확인(열 이름·행 수·작가 분포) + 팀 공통 폴드 파일 outputs/folds.csv 생성.

  python src/make_folds.py            # 없으면 생성, 있으면 현재 데이터와 일치하는지 검사
  python src/make_folds.py --remake   # 다시 생성 (데이터가 바뀐 경우. 팀원에게 다시 공유해야 함)
"""
from __future__ import annotations

import argparse

from config import LABEL_COL, add_common_args, get_paths
from data import describe, load_data, load_or_make_folds


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--remake", action="store_true")
    args = ap.parse_args()
    paths = get_paths(args.data, args.sample)
    train, test, sub = load_data(args.data, args.sample)
    describe(train, test, sub)
    folds = load_or_make_folds(train, paths.folds, remake=args.remake)
    print(f"\n[folds] {paths.folds}  ({len(folds):,}행)")
    tab = folds.groupby(["fold", LABEL_COL]).size().unstack(fill_value=0)
    tab["합계"] = tab.sum(axis=1)
    print("폴드별 작가 수 (층화 확인):")
    print(tab.to_string())


if __name__ == "__main__":
    main()
