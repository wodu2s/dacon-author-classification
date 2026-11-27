# 수업 범위 모델 비교와 최종 모델

- 기준: 전체 54,728행, 같은 5-fold(`outputs/folds.csv`) OOF. 주 지표 OOF logloss(낮을수록 좋음).
- 수업 범위: 단어 TF-IDF (1,2)까지 · 나이브 베이즈(+확률 보정) · 로지스틱 회귀 · 배깅 · 투표(단순 평균) · 부스팅(AdaBoost)
- 모든 모델의 입력은 TF-IDF 단어 (1,2)-gram(min_df=2), 벡터라이저는 폴드 학습 부분에만 fit

## ① 방식별 최고

| 방식 | 실험이름 | 설정 | OOF logloss | 정확도 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| 단일 NB | `mnb_tfidf_word12_a0.03_caliso` | MultinomialNB α=0.03 + isotonic 확률 보정 (NB 단계 결과 재사용) | 0.4725 | 83.26% | 82.57% |
| 단일 LR | `lr_tfidf_word12_C16` | 로지스틱 회귀 C=16 | 0.4603 | 84.01% | 83.26% |
| 배깅 | `bag10_lr_tfidf_word12_C16` | 로지스틱 회귀(C=16) × 10, 80% 복원추출 | 0.4894 | 83.34% | 82.53% |
| 부스팅 | `ada50_lr0.5_mnb_tfidf_word12_a0.03` | AdaBoost(MultinomialNB α=0.03), n_estimators=50, learning_rate=0.5 | 1.5763 | 64.94% | 61.36% |
| 투표 **(최종)** | `vote_mean_nb-lrw` | 단순 평균: NB(isotonic 보정) + 로지스틱 회귀(C=16) | 0.4207 | 86.05% | 85.43% |

## ② 전체 실험 (13개)

| 방식 | 실험이름 | 설정 | OOF logloss | 정확도 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| 단일 NB | `mnb_tfidf_word12_a0.03_caliso` | MultinomialNB α=0.03 + isotonic 확률 보정 (NB 단계 결과 재사용) | 0.4725 | 83.26% | 82.57% |
| 단일 NB | `mnb_tfidf_word12_a0.03` | MultinomialNB α=0.03, 보정 없음 (배깅·부스팅 비교 기준) | 0.4900 | 82.91% | 82.19% |
| 단일 LR | `lr_tfidf_word12_C1` | 로지스틱 회귀 C=1 | 0.6180 | 81.97% | 81.01% |
| 단일 LR | `lr_tfidf_word12_C4` | 로지스틱 회귀 C=4 | 0.4997 | 83.82% | 83.05% |
| 단일 LR | `lr_tfidf_word12_C16` | 로지스틱 회귀 C=16 | 0.4603 | 84.01% | 83.26% |
| 배깅 | `bag10_mnb_tfidf_word12_a0.03` | MultinomialNB(α=0.03) × 10, 80% 복원추출 | 0.5082 | 82.76% | 82.01% |
| 배깅 | `bag10_lr_tfidf_word12_C16` | 로지스틱 회귀(C=16) × 10, 80% 복원추출 | 0.4894 | 83.34% | 82.53% |
| 부스팅 | `ada50_lr0.5_mnb_tfidf_word12_a0.03` | AdaBoost(MultinomialNB α=0.03), n_estimators=50, learning_rate=0.5 | 1.5763 | 64.94% | 61.36% |
| 부스팅 | `ada50_lr1_mnb_tfidf_word12_a0.03` | AdaBoost(MultinomialNB α=0.03), n_estimators=50, learning_rate=1 | 1.5863 | 62.53% | 60.39% |
| 부스팅 | `ada100_lr0.5_mnb_tfidf_word12_a0.03` | AdaBoost(MultinomialNB α=0.03), n_estimators=100, learning_rate=0.5 | 1.5778 | 70.90% | 69.16% |
| 부스팅 | `ada100_lr1_mnb_tfidf_word12_a0.03` | AdaBoost(MultinomialNB α=0.03), n_estimators=100, learning_rate=1 | 1.5891 | 68.87% | 67.31% |
| 투표 | **`vote_mean_nb-lrw`** | 단순 평균: NB(isotonic 보정) + 로지스틱 회귀(C=16) | **0.4207** | 86.05% | 85.43% |
| 투표 | `vote_mean_nb-lrw-ada` | 단순 평균: NB(isotonic 보정) + 로지스틱 회귀(C=16) + AdaBoost(n=50, lr=0.5) | 0.6401 | 86.02% | 85.40% |

## ③ 최종 모델

| 항목 | 내용 |
|---|---|
| 모델 | vote_mean_nb-lrw — 투표 |
| 설정 | 단순 평균: NB(isotonic 보정) + 로지스틱 회귀(C=16) |
| 구성 모델 | mnb_tfidf_word12_a0.03_caliso (logloss 0.4725) + lr_tfidf_word12_C16 (logloss 0.4603) |
| 성능 (OOF) | logloss 0.4207 · 정확도 86.05% · Macro-F1 85.43% |
| NB 대비 | NB 최고(보정) logloss 0.4725 → 0.4207 (-0.0518), 정확도 83.26% → 86.05% (+2.79%p) |
| 선택 이유 | 수업 범위 실험 13개 중 OOF logloss(대회 평가지표)가 가장 낮음, 정확도도 가장 높음 |
| 제출 파일 | outputs/submissions/final_lecture.csv (스태킹 ensemble_best.csv, NB nb_best.csv 는 그대로 둠) |

## 참고

- **평가 기준**: 전체 학습 데이터 54,728행을 같은 5-fold(outputs/folds.csv)로 나눈 OOF. 최종 모델은 logloss 로 고름.
- **AdaBoost 확률**: sklearn 의 AdaBoost(SAMME)는 NB 의 확률이 아니라 예측 작가의 가중 투표를 softmax(f/4) 해서 확률을 낸다. 그래서 확률이 최대 0.255 라 logloss 가 1.368 아래로 내려갈 수 없다. 실제 OOF logloss 1.5763~1.5891, 정확도 62.53%~70.90% (단일 NB 보정 없음: 0.4900, 82.91%).
- **제외한 실험**: 수업 범위 밖 참고 실험 14개(선형 SVM, 글자 n-gram, 최적 가중치 voting, 스태킹)는 이 표와 최종 선택에서 뺌.
