"""결과 정리: logloss 상위 5개 표, 단계별 요약(results/nb_summary.md), alpha-logloss 그래프.

  python src/report.py
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from config import ALPHAS, add_common_args, get_paths
from experiments import Exp, stage1_exps

SHOW_COLS = ["exp", "model", "vectorizer", "analyzer", "ngram", "alpha", "calibration", "prep",
             "logloss", "logloss_testlike", "accuracy", "macro_f1", "time_sec"]
KOR = {"exp": "실험이름", "model": "모델", "vectorizer": "벡터라이저", "analyzer": "단위", "ngram": "n-gram",
       "alpha": "alpha", "calibration": "보정", "prep": "전처리", "logloss": "logloss",
       "logloss_testlike": "긴문장 logloss", "accuracy": "accuracy", "macro_f1": "macro_f1",
       "time_sec": "소요(s)", "logloss_fold_std": "폴드 std"}

# dataviz 기본 범주형 팔레트(라이트 모드), 고정 순서로 배정
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def config_label(row) -> str:
    short = {"MultinomialNB": "MNB", "BernoulliNB": "BNB", "CountVectorizer": "Count", "TfidfVectorizer": "TF-IDF"}
    return f"{short[row['model']]} · {short[row['vectorizer']]} · {row['analyzer']} {row['ngram']}"


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("logloss", "logloss_testlike", "accuracy", "macro_f1", "logloss_fold_std"):
        if c in df:
            df[c] = df[c].map(lambda v: f"{v:.4f}")
    if "alpha" in df:
        df["alpha"] = df["alpha"].map(lambda v: f"{v:g}")
    return df


def print_top(res: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if res.empty:
        print("결과가 없습니다.")
        return res
    top = res.nsmallest(n, "logloss")[SHOW_COLS]
    print(f"\n===== OOF logloss 상위 {n}개 =====")
    print(_fmt(top).rename(columns=KOR).to_string(index=False))
    return top


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def alpha_table(res: pd.DataFrame) -> pd.DataFrame:
    """보정·전처리 없는 설정 중 alpha 를 3개 이상 돌린 설정의 (설정 × alpha) logloss 표."""
    base = res[(res["calibration"] == "none") & (res["prep"] == "keep")].copy()
    base["config"] = base.apply(config_label, axis=1)
    counts = base.groupby("config")["alpha"].nunique()
    base = base[base["config"].isin(counts[counts >= 3].index)]
    if base.empty:
        return pd.DataFrame()
    piv = base.pivot_table(index="config", columns="alpha", values="logloss")
    return piv.loc[piv.min(axis=1).sort_values().index]


def build_summary(res: pd.DataFrame) -> str:
    out = ["# 나이브 베이즈 실험 요약 (5-fold OOF 기준)", "",
           f"- 실험 수: {len(res)}개  |  주 지표: OOF logloss (낮을수록 좋음)",
           "- 긴문장 logloss: 테스트 최소 길이 이상인 학습 행만으로 계산한 OOF logloss (테스트 조건 근사, 참고용)", ""]
    names1 = {e.name for e in stage1_exps()}
    s1 = res[res["exp"].isin(names1)].sort_values("logloss")
    if len(s1):
        out += ["## 1단계: 모델 × 벡터라이저 × 단위 (alpha=1)", "",
                _md_table(_fmt(s1[["exp", "model", "vectorizer", "analyzer", "ngram", "logloss", "accuracy",
                                   "macro_f1", "n_features", "time_sec"]]).rename(columns=KOR)), ""]
    piv = alpha_table(res)
    if len(piv):
        t = piv.map(lambda v: "" if pd.isna(v) else f"{v:.4f}").reset_index()
        t.columns = ["설정"] + [f"α={a:g}" for a in piv.columns]
        out += ["## 2단계: alpha(라플라스 스무딩) 탐색 — OOF logloss", "", _md_table(t), ""]
    plain = res[res["calibration"] == "none"]
    multi_prep = plain.copy()
    multi_prep["base"] = multi_prep["exp"].map(lambda n: replace(Exp.from_name(n), prep="keep").name)
    groups = multi_prep.groupby("base")["prep"].nunique()
    mp = multi_prep[multi_prep["base"].isin(groups[groups >= 2].index)]
    if len(mp):
        t = mp.pivot_table(index="base", columns="prep", values="logloss")
        t = t[[p for p in ["keep", "nostop", "nopunct", "nostopnopunct"] if p in t.columns]]
        t = t.map(lambda v: "" if pd.isna(v) else f"{v:.4f}").reset_index().rename(columns={"base": "설정"})
        out += ["## 3단계: 전처리 변형 (keep=유지 / nostop=불용어 제거 / nopunct=문장부호 제거) — OOF logloss",
                "", _md_table(t), ""]
    cal = res[res["calibration"] != "none"]
    if len(cal):
        rows = []
        for n in cal["exp"]:
            raw = replace(Exp.from_name(n), calib="none").name
            rows += [raw, n]
        t = res.set_index("exp").loc[list(dict.fromkeys(rows))].reset_index()
        out += ["## 4단계: 확률 보정 (CalibratedClassifierCV)", "",
                _md_table(_fmt(t[["exp", "calibration", "logloss", "logloss_testlike", "accuracy", "macro_f1",
                                  "time_sec"]]).rename(columns=KOR)), ""]
    top = res.nsmallest(5, "logloss")[SHOW_COLS]
    out += ["## OOF logloss 상위 5개", "", _md_table(_fmt(top).rename(columns=KOR)), ""]
    return "\n".join(out)


def plot_alpha(res: pd.DataFrame, path) -> bool:
    """alpha 에 따른 OOF logloss 변화 (모델별 패널, 설정별 선). 보고서용 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = alpha_table(res)
    if piv.empty:
        print("alpha 그래프: alpha 를 3개 이상 돌린 설정이 없어 생략")
        return False
    configs = list(piv.index)                       # 최저 logloss 순 → 색은 설정(entity)에 고정
    color = {c: SERIES_COLORS[i % len(SERIES_COLORS)] for i, c in enumerate(configs)}
    models = [m for m in ("MNB", "BNB") if any(c.startswith(m) for c in configs)]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    best_cfg, best_alpha = piv.stack().idxmin()
    for ax, m in zip(axes[0], models):
        ax.set_facecolor(SURFACE)
        for c in [c for c in configs if c.startswith(m)]:
            s = piv.loc[c].dropna()
            ax.plot(s.index, s.values, color=color[c], lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=c, zorder=3)
            a_min = s.idxmin()
            ax.annotate(f"{s.min():.3f}", (a_min, s.min()), textcoords="offset points", xytext=(0, 9),
                        ha="center", va="bottom", fontsize=8.5, color=INK_2)
        if best_cfg.startswith(m):
            ax.scatter([best_alpha], [piv.loc[best_cfg, best_alpha]], s=160, facecolors="none",
                       edgecolors=INK, linewidths=1.5, zorder=4, label="best (lowest OOF logloss)")
        ax.set_xscale("log")
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
        ax.minorticks_off()
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("alpha (Laplace smoothing, log scale)")
        ax.set_ylabel("OOF log loss (lower is better)")
        ax.set_title({"MNB": "MultinomialNB", "BNB": "BernoulliNB"}[m], color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2)   # 선·라벨을 가리지 않도록 그래프 아래에
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.06)   # 최저점 위 값 라벨 공간
    fig.suptitle("Naive Bayes: OOF log loss vs alpha (5-fold CV)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"alpha 그래프 저장 → {path}")
    return True


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    paths = get_paths(args.data, args.sample)
    res = pd.read_csv(paths.results_csv)
    print_top(res, 5)
    piv = alpha_table(res)
    if len(piv):
        print("\n===== alpha 탐색 (OOF logloss) =====")
        print(piv.map(lambda v: f"{v:.4f}" if not np.isnan(v) else "").to_string())
    paths.summary_md.write_text(build_summary(res), encoding="utf-8")
    print(f"\n요약 저장 → {paths.summary_md}")
    plot_alpha(res, paths.alpha_plot)


if __name__ == "__main__":
    main()
