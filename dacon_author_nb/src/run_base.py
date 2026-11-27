"""4-1. 기본 모델(앙상블 재료) 학습: NB 와 같은 폴드·같은 형식으로 OOF/테스트 확률 생성.

  python src/run_base.py                      # 로지스틱 회귀(단어·글자 × C) + 선형 SVM(× C) 전부
  python src/run_base.py --only lr_word svm   # 일부만
  python src/run_base.py --sample 2000        # 2,000행으로 빠른 흐름 점검 (sample_run_2000/ 에 따로 저장)

나이브 베이즈는 NB 단계 최고 설정(config.NB_BEST)의 OOF/테스트 확률을 그대로 재사용한다 (다시 학습하지 않음).
"""
from __future__ import annotations

import argparse
import time

from config import CHAR_NGRAM, ENS_N_JOBS, LR_CS, NB_BEST, SVM_CS, add_common_args
from ensemble_common import BaseSpec, add_n_jobs_args, load_ens_results, load_pred, nb_logloss, run_specs, setup

GROUPS = {
    # 로지스틱 회귀: TF-IDF 단어 (1,2)-gram — NB 최고 설정과 같은 특징
    "lr_word": [BaseSpec("lr", "word", (1, 2), C=c) for c in LR_CS],
    # 로지스틱 회귀: TF-IDF 글자 n-gram — 철자·어미·문장부호 습관 (단어와 다른 관점 → 앙상블 다양성)
    "lr_char": [BaseSpec("lr", "charwb", CHAR_NGRAM, C=c) for c in LR_CS],
    # 선형 SVM + sigmoid 확률 보정: TF-IDF 단어 (1,2)-gram
    "svm": [BaseSpec("svm", "word", (1, 2), C=c) for c in SVM_CS],
}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), help="실행할 묶음만 (기본: 전부)")
    ap.add_argument("--exp", nargs="+", help="실험 이름으로 지정 (예: svm_tfidf_word12_C0.1_calsig). --force 와 함께 재실행")
    add_n_jobs_args(ap, ENS_N_JOBS, "폴드 병렬 프로세스 수 (글자 n-gram 은 폴드마다 메모리를 많이 씀)")
    args = ap.parse_args()

    ctx = setup(args.data, args.sample)
    load_pred(ctx, NB_BEST)   # NB 최고 설정 OOF 가 있는지 먼저 확인 (앙상블 재료)
    print(f"[NB 재사용] {NB_BEST}  OOF logloss {nb_logloss(ctx, NB_BEST)}")

    t0 = time.perf_counter()
    if args.exp:
        run_specs([BaseSpec.from_name(n) for n in args.exp], ctx, n_jobs=args.n_jobs, force=args.force)
    for g in ([] if args.exp and not args.only else args.only or list(GROUPS)):
        print(f"\n========== {g}: 실험 {len(GROUPS[g])}개 ==========")
        run_specs(GROUPS[g], ctx, n_jobs=args.n_jobs, force=args.force)

    res = load_ens_results(ctx.paths)
    single = res[res["method"] == "단일"].sort_values("logloss")
    print(f"\n총 소요 {time.perf_counter() - t0:.0f}s  |  결과표: {ctx.paths.ens_results_csv}")
    print(single[["exp", "logloss", "accuracy", "macro_f1", "time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
