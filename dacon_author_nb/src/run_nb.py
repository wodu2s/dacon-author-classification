"""나이브 베이즈 실험 실행.

예)
  python src/run_nb.py --stage all                    # 1→2→3→4단계를 차례로 (앞 단계 결과로 다음 단계 후보를 고름)
  python src/run_nb.py --stage 1                      # 1단계만
  python src/run_nb.py --exp mnb_count_word12_a0.1    # 실험 하나(또는 여러 개)를 이름으로 지정
  python src/run_nb.py --stage all --sample 2000      # 2,000행으로 빠른 파이프라인 점검
"""
from __future__ import annotations

import argparse
import sys
import time

from config import N_JOBS, add_common_args, get_paths
from cv import load_results, run_experiments
from data import describe, load_data, load_or_make_folds
from experiments import STAGES, Exp
from report import print_top


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--stage", nargs="+", help="실행할 단계: 1 2 3 4 또는 all")
    ap.add_argument("--exp", nargs="+", help="실행할 실험 이름 (예: mnb_count_word12_a0.1)")
    ap.add_argument("--force", action="store_true", help="이미 끝난 실험도 다시 실행")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS, help="폴드 병렬 프로세스 수 (기본 %(default)s)")
    args = ap.parse_args()
    if not args.stage and not args.exp:
        ap.print_help()
        sys.exit(1)

    paths = get_paths(args.data, args.sample)
    train, test, sub = load_data(args.data, args.sample)
    describe(train, test, sub)
    folds = load_or_make_folds(train, paths.folds)
    print(f"[출력 폴더] {paths.root}")

    t0 = time.perf_counter()
    if args.exp:
        run_experiments([Exp.from_name(n) for n in args.exp], train, test, folds, paths, args.data,
                        args.n_jobs, args.force)
    if args.stage:
        stages = sorted(STAGES) if "all" in args.stage else sorted(int(s) for s in args.stage)
        for s in stages:
            exps = STAGES[s](load_results(paths.results_csv)) if s > 1 else STAGES[s]()
            print(f"\n========== {s}단계: 실험 {len(exps)}개 ==========")
            run_experiments(exps, train, test, folds, paths, args.data, args.n_jobs, args.force)

    print(f"\n총 소요 {time.perf_counter() - t0:.0f}s  |  결과표: {paths.results_csv}")
    print_top(load_results(paths.results_csv), 5)


if __name__ == "__main__":
    main()
