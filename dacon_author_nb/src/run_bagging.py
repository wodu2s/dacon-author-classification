"""4-2. 배깅(Bagging): 같은 모델을 부트스트랩 표본마다 따로 학습하고 확률을 평균 → 분산 감소 효과 확인.

  python src/run_bagging.py                              # MNB 배깅 + LR(최고 C) 배깅
  python src/run_bagging.py --only lr --n-estimators 5   # LR 배깅만, 모델 수를 줄여서 (시간 절약, 결과표에 기록됨)
  python src/run_bagging.py --lr-exp lr_tfidf_word12_C4  # 배깅할 LR 설정 지정 (기본: OOF logloss 최저 LR)

비교 기준(단일 모델):
  - MNB 배깅 ↔ NB 단계의 보정 없는 같은 설정 (config.NB_PLAIN, alpha = config.NB_BEST_ALPHA)
  - LR 배깅  ↔ run_base.py 의 같은 C 단일 LR
"""
from __future__ import annotations

import argparse
import time

from config import BAG_N_ESTIMATORS, NB_BEST_ALPHA, NB_PLAIN
from config import add_common_args
from ensemble_common import BaseSpec, add_n_jobs_args, load_ens_results, nb_logloss, run_specs, setup


def best_lr(ctx) -> str:
    res = load_ens_results(ctx.paths)
    lr = res[res["exp"].str.match(r"^lr_tfidf_")]
    if lr.empty:
        raise SystemExit("LR 결과가 없습니다 → 먼저 python src/run_base.py 실행")
    return lr.nsmallest(1, "logloss")["exp"].iloc[0]


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--only", nargs="+", choices=["mnb", "lr"], help="실행할 배깅만 (기본: 둘 다)")
    ap.add_argument("--lr-exp", help="배깅할 LR 실험 이름 (기본: OOF logloss 최저 LR)")
    ap.add_argument("--n-estimators", type=int, default=BAG_N_ESTIMATORS,
                    help="배깅 모델 수 (기본 %(default)s, 줄이면 결과표 note 에 기록)")
    # 배깅은 폴드를 차례로 돌리고, 배깅 안의 모델 10개를 병렬로 학습한다 (n_jobs=-1: 모든 코어)
    add_n_jobs_args(ap, -1, "배깅 내부 병렬 수 (BaggingClassifier n_jobs)")
    args = ap.parse_args()

    ctx = setup(args.data, args.sample)
    n = args.n_estimators
    specs = []
    if not args.only or "mnb" in args.only:
        specs.append(BaseSpec("mnb", "word", (1, 2), alpha=NB_BEST_ALPHA, n_estimators=n))
    if not args.only or "lr" in args.only:
        lr = BaseSpec.from_name(args.lr_exp or best_lr(ctx))
        specs.append(BaseSpec("lr", lr.analyzer, lr.ngram, C=lr.C, n_estimators=n))
    note = {s.name: f"n_estimators {BAG_N_ESTIMATORS}→{n} 으로 줄임(실행 시간)" for s in specs} \
        if n != BAG_N_ESTIMATORS else None

    t0 = time.perf_counter()
    for s in specs:
        run_specs([s], ctx, n_jobs=1, inner_jobs=args.n_jobs, force=args.force, extra_note=note)
    print(f"\n총 소요 {time.perf_counter() - t0:.0f}s")

    # 단일 모델 대비 비교 (logloss 가 줄었으면 분산 감소 효과)
    res = load_ens_results(ctx.paths).set_index("exp")
    print("\n===== 배깅 vs 단일 모델 (OOF logloss) =====")
    for s in specs:
        base = s.base_name
        single = nb_logloss(ctx, NB_PLAIN) if base == NB_PLAIN else (
            float(res.loc[base, "logloss"]) if base in res.index else None)
        bag = float(res.loc[s.name, "logloss"])
        diff = f"{bag - single:+.4f}" if single is not None else "?"
        print(f"  {base:<32} 단일 {single if single is None else f'{single:.4f}'}  →  "
              f"배깅 {bag:.4f}  (차이 {diff})")


if __name__ == "__main__":
    main()
