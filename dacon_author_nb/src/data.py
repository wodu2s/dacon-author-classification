"""데이터 로드, 팀 공통 폴드(StratifiedKFold), 텍스트 전처리 변형(불용어·문장부호 제거)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import CLASSES, DATASETS, ID_COL, LABEL_COL, N_FOLDS, SEED, TEXT_COL


def load_data(data_version: str, sample: int | None = None):
    """학습/테스트/제출양식을 읽고 기본 검사를 한다.

    학습 행 순서는 원본 index 오름차순으로 고정한다 → folds.csv, OOF(.npy) 행 순서의 기준.
    테스트 행 순서는 파일 그대로(= sample_submission 순서) 유지한다.
    """
    cfg = DATASETS[data_version]
    files = [*cfg["train_files"], cfg["test_file"], cfg["sample_submission"]]
    missing = [str(f) for f in files if not Path(f).exists()]
    if missing:
        raise SystemExit("데이터 파일이 없습니다:\n  " + "\n  ".join(missing)
                         + "\n→ src/config.py 의 DATASETS 경로를 확인하세요.")

    train = pd.concat([pd.read_csv(f) for f in cfg["train_files"]], ignore_index=True)
    test = pd.read_csv(cfg["test_file"])
    sub = pd.read_csv(cfg["sample_submission"])

    for col in (ID_COL, TEXT_COL, LABEL_COL):
        assert col in train.columns, f"학습 데이터에 '{col}' 열이 없음: {list(train.columns)}"
    for col in (ID_COL, TEXT_COL):
        assert col in test.columns, f"테스트 데이터에 '{col}' 열이 없음: {list(test.columns)}"

    train = train.sort_values(ID_COL).reset_index(drop=True)
    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    test[TEXT_COL] = test[TEXT_COL].fillna("").astype(str)
    train[LABEL_COL] = train[LABEL_COL].astype(int)

    assert train[ID_COL].is_unique and test[ID_COL].is_unique, "index 중복"
    assert set(train[LABEL_COL].unique()) <= set(CLASSES), "라벨이 0~4 범위를 벗어남"
    assert test[ID_COL].to_numpy().tolist() == sub[ID_COL].to_numpy().tolist(), \
        "테스트 행 순서가 sample_submission 과 다름"
    class_cols = [c for c in sub.columns if c != ID_COL]
    assert sorted(int(c) for c in class_cols) == CLASSES, f"제출 양식 열 이상: {class_cols}"

    if sample:
        # 빠른 파이프라인 점검용: 작가 비율을 유지한 채 N행만 사용 (테스트는 전체 유지 → 제출 형식까지 점검)
        train, _ = train_test_split(train, train_size=sample, stratify=train[LABEL_COL], random_state=SEED)
        train = train.sort_values(ID_COL).reset_index(drop=True)
    return train, test, sub


def describe(train: pd.DataFrame, test: pd.DataFrame, sub: pd.DataFrame) -> None:
    """열 이름·행 수·클래스 분포·문장 길이를 출력한다."""
    print(f"[train] {train.shape[0]:,}행  열={list(train.columns)}")
    print(f"[test ] {test.shape[0]:,}행  열={list(test.columns)}")
    print(f"[sample_submission] {sub.shape[0]:,}행  열={list(sub.columns)}")
    dist = train[LABEL_COL].value_counts().sort_index()
    print("[작가 분포] " + "  ".join(f"{k}: {v:,} ({v / len(train):.1%})" for k, v in dist.items()))
    tl, sl = train[TEXT_COL].str.len(), test[TEXT_COL].str.len()
    print(f"[문장 길이(글자)] train 평균 {tl.mean():.0f} / 중앙 {tl.median():.0f}  |  "
          f"test 평균 {sl.mean():.0f} / 중앙 {sl.median():.0f} / 최소 {sl.min()}")
    print(f"  → 테스트 최소 길이({sl.min()}자) 이상인 학습 행: {(tl >= sl.min()).mean():.1%}")


# ---------------------------------------------------------------------------
# 팀 공통 검증 틀: StratifiedKFold(5, shuffle=True, random_state=42)
# ---------------------------------------------------------------------------
def make_folds(train: pd.DataFrame) -> pd.DataFrame:
    """층화 K-겹 교차검증: 폴드마다 작가 비율이 전체와 같도록 나눈다."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(train), -1, dtype=int)
    for k, (_, va_idx) in enumerate(skf.split(train[TEXT_COL], train[LABEL_COL])):
        fold[va_idx] = k
    assert (fold >= 0).all()
    return pd.DataFrame({ID_COL: train[ID_COL].to_numpy(), LABEL_COL: train[LABEL_COL].to_numpy(), "fold": fold})


def load_or_make_folds(train: pd.DataFrame, path: Path, remake: bool = False) -> pd.DataFrame:
    """folds.csv 가 있으면 읽어서 현재 데이터와 행이 일치하는지 검사, 없으면 만들어 저장한다."""
    if path.exists() and not remake:
        folds = pd.read_csv(path)
        same = (len(folds) == len(train)
                and (folds[ID_COL].to_numpy() == train[ID_COL].to_numpy()).all()
                and (folds[LABEL_COL].to_numpy() == train[LABEL_COL].to_numpy()).all())
        if not same:
            raise SystemExit(f"{path} 의 행(index/author)이 현재 학습 데이터와 다릅니다. "
                             "데이터가 바뀌었다면 make_folds.py --remake 로 다시 만드세요 (팀원과 공유 필수).")
        return folds
    folds = make_folds(train)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    print(f"[folds] 새로 생성 → {path}")
    return folds


# ---------------------------------------------------------------------------
# 전처리 변형 (3단계 비교용). 행 단위 규칙 기반 변환이라 학습 통계를 쓰지 않음 → 데이터 누수 없음
#   keep          : 그대로 (불용어·문장부호 유지)
#   nostop        : 불용어 제거 (sklearn 영어 불용어 목록)
#   nopunct       : 문장부호 제거
#   nostopnopunct : 둘 다 제거
# 작가 분류에서는 기능어(the, of, and...)·문장부호(따옴표 모양, ; — 등)가 문체 신호라 제거가 손해일 수 있다.
# ---------------------------------------------------------------------------
_STOP_RE = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, ENGLISH_STOP_WORDS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")


def apply_prep(texts: pd.Series, prep: str) -> pd.Series:
    if prep == "keep":
        return texts
    s = texts
    if prep in ("nostop", "nostopnopunct"):
        s = s.str.replace(_STOP_RE, " ", regex=True)
    if prep in ("nopunct", "nostopnopunct"):
        s = s.str.replace(_PUNCT_RE, " ", regex=True)
    if prep not in ("nostop", "nopunct", "nostopnopunct"):
        raise ValueError(f"알 수 없는 전처리 변형: {prep}")
    return s.str.replace(r"\s+", " ", regex=True).str.strip()
