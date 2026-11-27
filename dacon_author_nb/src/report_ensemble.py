"""4-5. 앙상블 결과 정리: NB 최고 결과와 방식별 최고 결과를 한 표로 비교 (보고서용).

  python src/report_ensemble.py
    → results/ensemble_summary.md   (방식별 최고 / 기본 모델 / 배깅 vs 단일 / soft voting / 스태킹 표)
    → results/ensemble_compare.png  (방식별 최고 OOF logloss 막대그래프)
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import NB_BEST, add_common_args, get_paths
from ensemble_common import METHOD_BAG, METHOD_SINGLE, METHOD_STACK, METHOD_VOTE, BaseSpec
from report import GRID, INK, INK_2, SERIES_COLORS, SURFACE, _md_table

METHOD_ORDER = [METHOD_SINGLE, METHOD_BAG, METHOD_VOTE, METHOD_STACK]
GROUP_KOR = {"NB": "나이브 베이즈 (NB 단계 최고)", METHOD_SINGLE: "단일 모델 (LR / SVM)", METHOD_BAG: "배깅",
             METHOD_VOTE: "soft voting", METHOD_STACK: "스태킹 (부스팅 메타 포함)"}
GROUP_EN = {"NB": "Naive Bayes (single)", METHOD_SINGLE: "Single LR / SVM", METHOD_BAG: "Bagging",
            METHOD_VOTE: "Soft voting", METHOD_STACK: "Stacking"}
COLS_KOR = {"exp": "실험이름", "logloss": "logloss", "accuracy": "accuracy", "macro_f1": "macro_f1",
            "time_sec": "소요(s)", "logloss_testlike": "긴문장 logloss", "settings": "설정", "note": "비고"}


def _f4(df, cols=("logloss", "accuracy", "macro_f1", "logloss_testlike")):
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    if "note" in df:
        df["note"] = df["note"].fillna("")
    return df


def comparison(nb: pd.DataFrame, ens: pd.DataFrame) -> pd.DataFrame:
    """방식별 OOF logloss 최저 실험 1개씩 (+ NB 단계 최고)."""
    rows = []
    if NB_BEST in set(nb["exp"]):
        r = nb.set_index("exp").loc[NB_BEST]
        rows.append(dict(group="NB", exp=NB_BEST, logloss=r["logloss"], accuracy=r["accuracy"],
                         macro_f1=r["macro_f1"], logloss_testlike=r["logloss_testlike"], time_sec=r["time_sec"]))
    for m in METHOD_ORDER:
        sub = ens[ens["method"] == m]
        if len(sub):
            r = sub.nsmallest(1, "logloss").iloc[0]
            rows.append(dict(group=m, exp=r["exp"], logloss=r["logloss"], accuracy=r["accuracy"],
                             macro_f1=r["macro_f1"], logloss_testlike=r["logloss_testlike"], time_sec=r["time_sec"]))
    return pd.DataFrame(rows)


def bagging_table(nb: pd.DataFrame, ens: pd.DataFrame) -> pd.DataFrame:
    """배깅 vs 같은 설정의 단일 모델 (logloss 가 줄면 분산 감소 효과)."""
    single = pd.concat([nb[["exp", "logloss", "accuracy"]], ens[["exp", "logloss", "accuracy"]]]).set_index("exp")
    rows = []
    for _, r in ens[ens["method"] == METHOD_BAG].iterrows():
        base = BaseSpec.from_name(r["exp"]).base_name
        s = single.loc[base] if base in single.index else None
        rows.append({"단일 모델": base, "단일 logloss": "" if s is None else f"{s['logloss']:.4f}",
                     "배깅 모델": r["exp"], "배깅 logloss": f"{r['logloss']:.4f}",
                     "차이(배깅-단일)": "" if s is None else f"{r['logloss'] - s['logloss']:+.4f}",
                     "단일 acc": "" if s is None else f"{s['accuracy']:.4f}", "배깅 acc": f"{r['accuracy']:.4f}",
                     "소요(s)": r["time_sec"], "비고": "" if pd.isna(r["note"]) else r["note"]})
    return pd.DataFrame(rows)


def build_summary(nb: pd.DataFrame, ens: pd.DataFrame) -> str:
    comp = comparison(nb, ens)
    t = _f4(comp).assign(group=comp["group"].map(GROUP_KOR)).rename(columns={"group": "방식", **COLS_KOR})
    out = ["# 앙상블 실험 요약 (5-fold OOF 기준, 같은 outputs/folds.csv)", "",
           "- 주 지표: OOF logloss (낮을수록 좋음). 긴문장 logloss = 테스트 최소 길이 이상인 학습 행만의 OOF logloss (참고)",
           "- NB 행의 소요시간은 NB 단계(다른 PC, 24코어)에서 잰 값이라 앙상블 행과 직접 비교할 수 없다.", "",
           "## 방식별 최고 (NB 단계 최고와 비교)", "", _md_table(t), ""]
    cols = ["exp", "logloss", "logloss_testlike", "accuracy", "macro_f1", "time_sec", "settings", "note"]
    single = ens[ens["method"] == METHOD_SINGLE].sort_values("logloss")
    if len(single):
        out += ["## 4-1. 기본 모델 (단일)", "", _md_table(_f4(single[cols]).rename(columns=COLS_KOR)), ""]
    bag = bagging_table(nb, ens)
    if len(bag):
        out += ["## 4-2. 배깅 vs 단일 모델", "", _md_table(bag), ""]
    vote = ens[ens["method"] == METHOD_VOTE].sort_values("logloss")
    if len(vote):
        out += ["## 4-3. soft voting", "", _md_table(_f4(vote[["exp", "models", "logloss", "accuracy", "macro_f1",
                                                              "settings", "note"]]).rename(
            columns={**COLS_KOR, "models": "사용 모델"})), ""]
    st = ens[ens["method"] == METHOD_STACK].sort_values("logloss")
    if len(st):
        out += ["## 4-4. 스태킹 (메타 모델: LightGBM 부스팅 / 로지스틱 회귀)", "",
                _md_table(_f4(st[cols]).rename(columns=COLS_KOR)), ""]
    return "\n".join(out)


def plot_compare(comp: pd.DataFrame, path) -> None:
    """방식별 최고 OOF logloss 가로 막대그래프 (단일 시리즈 → 한 색, 값은 막대 끝에 직접 표기)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK_2})
    comp = comp.iloc[::-1].reset_index(drop=True)   # 위에서부터 NB → 단일 → 배깅 → voting → 스태킹
    labels = [f"{GROUP_EN[g]}\n{e}" for g, e in zip(comp["group"], comp["exp"])]
    best = comp["logloss"].idxmin()
    fig, ax = plt.subplots(figsize=(8.6, 0.78 * len(comp) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(range(len(comp)), comp["logloss"], height=0.56, color=SERIES_COLORS[0], zorder=3)
    hi = comp["logloss"].max()
    for i, v in enumerate(comp["logloss"]):
        ax.text(v + hi * 0.012, i, f"{v:.4f}" + ("  (best)" if i == best else ""), va="center", fontsize=9.5,
                color=INK if i == best else INK_2, fontweight="bold" if i == best else "normal")
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlim(0, hi * 1.22)
    ax.set_xlabel("OOF log loss (lower is better)")
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Best OOF log loss by method (5-fold CV, same folds)", color=INK, fontsize=11.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"그래프 저장 → {path}")


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    args = ap.parse_args()
    paths = get_paths(args.data, args.sample)
    nb = pd.read_csv(paths.results_csv) if paths.results_csv.exists() else pd.DataFrame(columns=["exp"])
    ens = pd.read_csv(paths.ens_results_csv)
    comp = comparison(nb, ens)
    print("===== 방식별 최고 OOF logloss (NB 단계 최고와 비교) =====")
    print(_f4(comp).assign(group=comp["group"].map(GROUP_KOR)).rename(columns={"group": "방식", **COLS_KOR})
          .to_string(index=False))
    bag = bagging_table(nb, ens)
    if len(bag):
        print("\n===== 배깅 vs 단일 =====")
        print(bag.drop(columns=["비고"]).to_string(index=False))
    paths.ens_summary_md.write_text(build_summary(nb, ens), encoding="utf-8")
    print(f"\n요약 저장 → {paths.ens_summary_md}")
    plot_compare(comp, paths.ens_plot)


if __name__ == "__main__":
    main()
