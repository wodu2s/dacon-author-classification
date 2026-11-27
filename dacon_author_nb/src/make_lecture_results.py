"""수업 범위 4-3·4-4. 수업에서 배운 방법만 모아 결과표·비교 문서·최종 제출 파일을 만든다.

  python src/make_lecture_results.py
    → results/lecture_scope_results.csv     방식 | 실험이름 | 설정 | OOF logloss | 정확도 | Macro-F1
    → results/lecture_scope_summary.md      ① 방식별 최고 ② 전체 실험 ③ 최종 모델
    → results/model_comparison_lecture.pdf  같은 내용 A4 한 장 (Windows 의 Microsoft Edge 로 인쇄, 없으면 건너뜀)
    → outputs/submissions/final_lecture.csv (+ .json)  최종 모델 = 이 표에서 OOF logloss 최저

수업 범위(사용): 단어 TF-IDF (n-gram (1,2)까지), 나이브 베이즈(+NB 확률 보정), 로지스틱 회귀,
               배깅(BaggingClassifier), 투표(soft voting 단순 평균), 부스팅(AdaBoostClassifier)
수업 범위 밖(참고 실험, 제외): 선형 SVM, 글자 n-gram, 최적 가중치(wopt) voting, 스태킹
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import ID_COL, N_CLASSES, NB_BEST, NB_PLAIN, add_common_args, get_paths
from data import load_data

METHOD_ORDER = ["단일 NB", "단일 LR", "배깅", "부스팅", "투표"]
LECTURE_VOTE_CODES = {"nb", "lrw", "ada"}
COLS = ["방식", "실험이름", "설정", "OOF logloss", "정확도", "Macro-F1"]


# ---------------------------------------------------------------------------
# 수업 범위 판정 + 짧은 설정 설명 (실험 이름 규칙에서 읽음)
# ---------------------------------------------------------------------------
def lecture_method(exp: str) -> str | None:
    if re.match(r"^lr_tfidf_word\d\d_C", exp):
        return "단일 LR"
    if re.match(r"^bag\d+_(mnb|lr)_tfidf_word", exp):
        return "배깅"
    if re.match(r"^ada\d+_", exp):
        return "부스팅"
    m = re.match(r"^vote_mean_(.+)$", exp)            # 단순 평균만 (vote_wopt = 최적 가중치는 제외)
    if m and set(m[1].split("-")) <= LECTURE_VOTE_CODES:
        return "투표"
    return None                                       # SVM, 글자 n-gram, wopt, 스태킹 → 참고 실험


def short_label(exp: str) -> str:
    """투표 구성원 등을 짧게 부를 이름."""
    if exp == NB_BEST:
        return "NB(isotonic 보정)"
    if m := re.match(r"^lr_tfidf_word12_C([\d.]+)$", exp):
        return f"로지스틱 회귀(C={m[1]})"
    if m := re.match(r"^ada(\d+)_lr([\d.]+)_", exp):
        return f"AdaBoost(n={m[1]}, lr={m[2]})"
    return exp


def describe(exp: str, models: str = "") -> str:
    """설정 설명. 모든 실험의 입력은 TF-IDF 단어 (1,2)-gram (min_df=2, sublinear_tf) 이라 생략."""
    if exp == NB_BEST:
        return "MultinomialNB α=0.03 + isotonic 확률 보정 (NB 단계 결과 재사용)"
    if exp == NB_PLAIN:
        return "MultinomialNB α=0.03, 보정 없음 (배깅·부스팅 비교 기준)"
    if m := re.match(r"^lr_tfidf_word(\d)(\d)_C([\d.]+)$", exp):
        return f"로지스틱 회귀 C={m[3]}"
    if m := re.match(r"^bag(\d+)_mnb_tfidf_word12_a([\d.]+)$", exp):
        return f"MultinomialNB(α={m[2]}) × {m[1]}, 80% 복원추출"
    if m := re.match(r"^bag(\d+)_lr_tfidf_word12_C([\d.]+)$", exp):
        return f"로지스틱 회귀(C={m[2]}) × {m[1]}, 80% 복원추출"
    if m := re.match(r"^ada(\d+)_lr([\d.]+)_mnb_tfidf_word12_a([\d.]+)$", exp):
        return f"AdaBoost(MultinomialNB α={m[3]}), n_estimators={m[1]}, learning_rate={m[2]}"
    if exp.startswith("vote_mean_"):
        members = [p.split("=", 1)[1].strip() for p in models.split(" + ")]
        return "단순 평균: " + " + ".join(short_label(e) for e in members)
    return exp


def build_table(paths) -> pd.DataFrame:
    nb = pd.read_csv(paths.results_csv).set_index("exp")
    ens = pd.read_csv(paths.ens_results_csv)
    rows = []
    for exp in (NB_BEST, NB_PLAIN):     # 단일 NB: NB 단계 결과 재사용 (다시 학습하지 않음)
        r = nb.loc[exp]
        rows.append(dict(방식="단일 NB", 실험이름=exp, 설정=describe(exp), logloss=r["logloss"],
                         accuracy=r["accuracy"], macro_f1=r["macro_f1"], models=""))
    for _, r in ens.iterrows():
        m = lecture_method(r["exp"])
        if m:
            rows.append(dict(방식=m, 실험이름=r["exp"], 설정=describe(r["exp"], r["models"]), logloss=r["logloss"],
                             accuracy=r["accuracy"], macro_f1=r["macro_f1"], models=r["models"]))
    df = pd.DataFrame(rows)
    df["_order"] = df["방식"].map(METHOD_ORDER.index)
    df = df.sort_values(["_order"], kind="stable").drop(columns="_order").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 제출 파일 + 검증 (저장한 파일을 다시 읽어서 확인)
# ---------------------------------------------------------------------------
def write_submission(paths, data_version: str, exp: str, row) -> Path:
    _, test, sub = load_data(data_version)
    pred = np.load(paths.test_dir / f"{exp}.npy")
    assert pred.shape == (len(sub), N_CLASSES), f"테스트 예측 shape 이상: {pred.shape}"
    pred = pred / pred.sum(axis=1, keepdims=True)        # 각 행 합 = 1
    out = sub.copy()
    for col in [c for c in sub.columns if c != ID_COL]:
        out[col] = pred[:, int(col)]                     # 열 "k" ← 작가 k 확률 (sample_submission 열 순서 유지)
    path = paths.sub_dir / "final_lecture.csv"
    out.to_csv(path, index=False)

    chk = pd.read_csv(path)
    prob = chk.drop(columns=ID_COL)
    checks = {
        "행 수 = sample_submission": len(chk) == len(sub),
        "열 이름·순서 = sample_submission": list(chk.columns) == list(sub.columns),
        "index 순서 = sample_submission": bool((chk[ID_COL].to_numpy() == sub[ID_COL].to_numpy()).all()),
        "결측값 없음": not chk.isna().any().any(),
        "확률 0~1 범위": bool(((prob >= 0) & (prob <= 1)).all().all()),
        "각 행 확률 합 = 1 (오차 1e-6)": bool(np.allclose(prob.sum(axis=1), 1.0, atol=1e-6)),
    }
    print(f"\n제출 파일 → {path}")
    for k, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {k}")
    assert all(checks.values()), "제출 파일 검증 실패"
    print(f"  shape {chk.shape}, 행 합 min {prob.sum(axis=1).min():.6f} / max {prob.sum(axis=1).max():.6f}, "
          f"확률 최솟값 {prob.min().min():.2e}, 예측 작가 분포 {np.bincount(prob.to_numpy().argmax(1), minlength=N_CLASSES).tolist()}")
    info = dict(exp=exp, method=row["방식"], settings=row["설정"], oof_logloss=float(row["logloss"]),
                oof_accuracy=float(row["accuracy"]), oof_macro_f1=float(row["macro_f1"]),
                test_pred=f"outputs/test_pred/{exp}.npy", scope="수업 범위 (투표·배깅·부스팅, 단어 TF-IDF·NB·로지스틱 회귀)",
                checks={k: bool(v) for k, v in checks.items()})
    (paths.sub_dir / "final_lecture.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 요약 문서 (md / PDF 공통 내용)
# ---------------------------------------------------------------------------
def pct(v):
    return f"{v * 100:.2f}%"


def summary_parts(df: pd.DataFrame, n_excluded: int) -> dict:
    best = {m: df[df["방식"] == m].nsmallest(1, "logloss").iloc[0] for m in METHOD_ORDER if (df["방식"] == m).any()}
    final = df.nsmallest(1, "logloss").iloc[0]
    nb = df.set_index("실험이름").loc[NB_BEST]
    top_acc = df.nlargest(1, "accuracy").iloc[0]
    reason = f"수업 범위 실험 {len(df)}개 중 OOF logloss(대회 평가지표)가 가장 낮음"
    reason += (", 정확도도 가장 높음" if top_acc["실험이름"] == final["실험이름"] else
               f" (정확도 최고는 {top_acc['실험이름']} {pct(top_acc['accuracy'])})")
    members = ""
    if final["방식"] == "투표":
        mem = [p.split("=", 1)[1].strip() for p in final["models"].split(" + ")]
        allrows = df.set_index("실험이름")
        members = " + ".join(f"{e} (logloss {allrows.loc[e, 'logloss']:.4f})" if e in allrows.index else e for e in mem)
    ada = df[df["방식"] == "부스팅"]
    nb_plain = df.set_index("실험이름").loc[NB_PLAIN]
    final_kv = [
        ("모델", f"{final['실험이름']} — {final['방식']}"),
        ("설정", final["설정"]),
    ]
    if members:
        final_kv.append(("구성 모델", members))
    final_kv += [
        ("성능 (OOF)", f"logloss {final['logloss']:.4f} · 정확도 {pct(final['accuracy'])} · Macro-F1 {pct(final['macro_f1'])}"),
        ("NB 대비", f"NB 최고(보정) logloss {nb['logloss']:.4f} → {final['logloss']:.4f} ({final['logloss'] - nb['logloss']:+.4f}), "
                   f"정확도 {pct(nb['accuracy'])} → {pct(final['accuracy'])} ({(final['accuracy'] - nb['accuracy']) * 100:+.2f}%p)"),
        ("선택 이유", reason),
        ("제출 파일", "outputs/submissions/final_lecture.csv (스태킹 ensemble_best.csv, NB nb_best.csv 는 그대로 둠)"),
    ]
    notes = [
        ("평가 기준", "전체 학습 데이터 54,728행을 같은 5-fold(outputs/folds.csv)로 나눈 OOF. 최종 모델은 logloss 로 고름."),
    ]
    if len(ada):
        notes.append(("AdaBoost 확률",
                      "sklearn 의 AdaBoost(SAMME)는 NB 의 확률이 아니라 예측 작가의 가중 투표를 softmax(f/4) 해서 확률을 낸다. "
                      f"그래서 확률이 최대 0.255 라 logloss 가 1.368 아래로 내려갈 수 없다. 실제 OOF logloss {ada['logloss'].min():.4f}~{ada['logloss'].max():.4f}, "
                      f"정확도 {pct(ada['accuracy'].min())}~{pct(ada['accuracy'].max())} (단일 NB 보정 없음: {nb_plain['logloss']:.4f}, {pct(nb_plain['accuracy'])})."))
    notes.append(("제외한 실험", f"수업 범위 밖 참고 실험 {n_excluded}개(선형 SVM, 글자 n-gram, 최적 가중치 voting, 스태킹)는 이 표와 최종 선택에서 뺌."))
    return dict(best=best, final=final, final_kv=final_kv, notes=notes)


def build_md(df: pd.DataFrame, parts: dict) -> str:
    def row(cells):
        return "| " + " | ".join(cells) + " |"
    hdr = ["방식", "실험이름", "설정", "OOF logloss", "정확도", "Macro-F1"]
    out = ["# 수업 범위 모델 비교와 최종 모델", "",
           "- 기준: 전체 54,728행, 같은 5-fold(`outputs/folds.csv`) OOF. 주 지표 OOF logloss(낮을수록 좋음).",
           "- 수업 범위: 단어 TF-IDF (1,2)까지 · 나이브 베이즈(+확률 보정) · 로지스틱 회귀 · 배깅 · 투표(단순 평균) · 부스팅(AdaBoost)",
           "- 모든 모델의 입력은 TF-IDF 단어 (1,2)-gram(min_df=2), 벡터라이저는 폴드 학습 부분에만 fit", "",
           "## ① 방식별 최고", "", row(hdr), row(["---"] * 6)]
    fin = parts["final"]["실험이름"]
    for m, r in parts["best"].items():
        mark = " **(최종)**" if r["실험이름"] == fin else ""
        out.append(row([m + mark, f"`{r['실험이름']}`", r["설정"], f"{r['logloss']:.4f}", pct(r["accuracy"]), pct(r["macro_f1"])]))
    out += ["", f"## ② 전체 실험 ({len(df)}개)", "", row(hdr), row(["---"] * 6)]
    for _, r in df.iterrows():
        b = "**" if r["실험이름"] == fin else ""
        out.append(row([r["방식"], f"{b}`{r['실험이름']}`{b}", r["설정"], f"{b}{r['logloss']:.4f}{b}", pct(r["accuracy"]), pct(r["macro_f1"])]))
    out += ["", "## ③ 최종 모델", "", "| 항목 | 내용 |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in parts["final_kv"]]
    out += ["", "## 참고", ""] + [f"- **{k}**: {v}" for k, v in parts["notes"]]
    return "\n".join(out) + "\n"


CSS = """
@page { size: A4 portrait; margin: 10mm 11mm 9mm 11mm; }
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --rule:#d9d8d3; --head:#f1f0ec; --accent:#2a78d6; --tint:#e8f1fc; }
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background:#fff; color:var(--ink); font-family:"Malgun Gothic","맑은 고딕",sans-serif; font-size:7.8pt; line-height:1.36; }
h1 { font-size:14pt; margin:0 0 1.2mm; }
.sub { color:var(--ink2); font-size:7.8pt; }
.basis { color:var(--ink2); font-size:7.4pt; padding:1.5mm 2.4mm; background:var(--head); border-radius:1.2mm; margin:1.8mm 0 2.6mm; }
h2 { font-size:9.6pt; margin:3mm 0 1.3mm; padding-left:1.8mm; border-left:0.8mm solid var(--accent); }
table { width:100%; border-collapse:collapse; }
th, td { padding:0.7mm 1.5mm; border-bottom:0.25mm solid var(--rule); vertical-align:top; text-align:left; }
th { background:var(--head); color:var(--ink2); font-size:7.4pt; border-bottom:0.35mm solid #bdbcb6; }
td:first-child, th:first-child { white-space:nowrap; }
td.n, th.n { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
code { font-family:Consolas,"Cascadia Mono",monospace; font-size:7.2pt; color:#263238; white-space:nowrap; }
tr.final td { background:var(--tint); font-weight:700; }
tr.gs td { border-top:0.35mm solid #bdbcb6; }
.badge { display:inline-block; font-size:6.5pt; font-weight:700; color:#fff; background:var(--accent); border-radius:0.8mm; padding:0 1.1mm; margin-left:1mm; }
.kv th { width:22mm; }
ul { margin:0.8mm 0 0; padding-left:4.2mm; } li { margin:0.6mm 0; }
.foot { margin-top:2.2mm; color:var(--muted); font-size:6.8pt; }
"""


def build_html(df: pd.DataFrame, parts: dict) -> str:
    e = html.escape
    fin = parts["final"]["실험이름"]

    def num_cells(r):
        return (f'<td class="n">{r["logloss"]:.4f}</td><td class="n">{pct(r["accuracy"])}</td>'
                f'<td class="n">{pct(r["macro_f1"])}</td>')
    head_num = '<th class="n">OOF logloss</th><th class="n">정확도</th><th class="n">Macro-F1</th>'
    t1 = [f'<table><thead><tr><th>방식</th><th>실험</th><th>설정</th>{head_num}</tr></thead><tbody>']
    for m, r in parts["best"].items():
        cls = ' class="final"' if r["실험이름"] == fin else ""
        badge = '<span class="badge">최종</span>' if cls else ""
        t1.append(f'<tr{cls}><td>{e(m)}{badge}</td><td><code>{e(r["실험이름"])}</code></td><td>{e(r["설정"])}</td>{num_cells(r)}</tr>')
    t1.append("</tbody></table>")
    t2 = [f'<table><thead><tr><th>방식</th><th>실험</th><th>설정</th>{head_num}</tr></thead><tbody>']
    prev = None
    for _, r in df.iterrows():
        cls = []
        if prev is not None and r["방식"] != prev:
            cls.append("gs")
        if r["실험이름"] == fin:
            cls.append("final")
        badge = '<span class="badge">최종</span>' if r["실험이름"] == fin else ""
        t2.append(f'<tr class="{" ".join(cls)}"><td>{e(r["방식"])}{badge}</td><td><code>{e(r["실험이름"])}</code></td>'
                  f'<td>{e(r["설정"])}</td>{num_cells(r)}</tr>')
        prev = r["방식"]
    t2.append("</tbody></table>")
    t3 = ['<table class="kv"><tbody>'] + [f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in parts["final_kv"]] + ["</tbody></table>"]
    notes = "".join(f"<li><b>{e(k)}:</b> {e(v)}</li>" for k, v in parts["notes"])
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>수업 범위 모델 비교</title>
<style>{CSS}</style></head><body>
<h1>소설 작가 분류 — 수업 범위 모델 비교와 최종 모델</h1>
<div class="sub">응용머신러닝(2026-2) 기말 팀 프로젝트 · 월간 데이콘 「소설 작가 분류 AI 경진대회」(235670) · 평가지표 multi-class logloss</div>
<div class="basis">기준: 전체 학습 데이터 54,728행을 같은 5-fold(<code>outputs/folds.csv</code>)로 나눈 <b>OOF</b> 성능, 선택 기준은 <b>logloss(낮을수록 좋음)</b>.
수업 범위: 단어 TF-IDF (1,2)까지 · 나이브 베이즈(+확률 보정) · 로지스틱 회귀 · 배깅 · 투표(단순 평균) · 부스팅(AdaBoost).
모든 모델의 입력은 TF-IDF 단어 (1,2)-gram(min_df=2), 벡터라이저는 폴드 학습 부분에만 fit. 출처: <code>results/lecture_scope_results.csv</code></div>
<h2>① 방식별 최고</h2>{''.join(t1)}
<h2>② 전체 실험 ({len(df)}개)</h2>{''.join(t2)}
<h2>③ 최종 모델</h2>{''.join(t3)}
<h2>참고</h2><ul>{notes}</ul>
<div class="foot">GitHub wodu2s/dacon-author-classification · ensemble 브랜치 <code>dacon_author_nb/</code> · <code>python src/make_lecture_results.py</code> 로 생성</div>
</body></html>"""


def print_pdf(html_text: str, pdf_path: Path) -> None:
    """Windows 의 Microsoft Edge(헤드리스)로 HTML → PDF. Edge 가 없으면 건너뜀 (md 는 그대로 생성됨)."""
    cands = [shutil.which("msedge"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
    edge = next((c for c in cands if c and Path(c).exists()), None)
    if not edge:
        print("PDF: Microsoft Edge 를 찾지 못해 건너뜀 (md 파일 참고)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "lecture.html"
        src.write_text(html_text, encoding="utf-8")
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={tmp}/edge",
                        f"--print-to-pdf={pdf_path}", src.as_uri()], check=True, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"PDF 저장 → {pdf_path} ({pages}페이지)")


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--no-pdf", action="store_true", help="PDF 생성 건너뛰기")
    args = ap.parse_args()
    paths = get_paths(args.data, args.sample)

    df = build_table(paths)
    ens = pd.read_csv(paths.ens_results_csv)
    n_excluded = int(sum(lecture_method(x) is None for x in ens["exp"]))
    out = df.rename(columns={"logloss": "OOF logloss", "accuracy": "정확도", "macro_f1": "Macro-F1"})[COLS]
    out.to_csv(paths.lecture_csv, index=False, encoding="utf-8-sig")   # 엑셀에서 한글이 깨지지 않도록 BOM 포함
    print(f"수업 범위 결과표 ({len(df)}개, 제외한 참고 실험 {n_excluded}개) → {paths.lecture_csv}")
    show = out.copy()
    for c in ("OOF logloss", "정확도", "Macro-F1"):
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    print(show.drop(columns="설정").to_string(index=False))

    parts = summary_parts(df, n_excluded)
    final = parts["final"]
    print(f"\n최종 모델 (OOF logloss 최저): {final['실험이름']} [{final['방식']}] logloss {final['logloss']:.4f}, "
          f"정확도 {final['accuracy']:.4f}, Macro-F1 {final['macro_f1']:.4f}")
    write_submission(paths, args.data, final["실험이름"], final)

    paths.lecture_md.write_text(build_md(df, parts), encoding="utf-8")
    print(f"요약 저장 → {paths.lecture_md}")
    if not args.no_pdf:
        print_pdf(build_html(df, parts), paths.lecture_pdf)


if __name__ == "__main__":
    main()
