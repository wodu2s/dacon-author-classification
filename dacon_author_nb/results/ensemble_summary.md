# 앙상블 실험 요약 (5-fold OOF 기준, 같은 outputs/folds.csv)

- 주 지표: OOF logloss (낮을수록 좋음). 긴문장 logloss = 테스트 최소 길이 이상인 학습 행만의 OOF logloss (참고)
- NB 행의 소요시간은 NB 단계(다른 PC, 24코어)에서 잰 값이라 앙상블 행과 직접 비교할 수 없다.

## 방식별 최고 (NB 단계 최고와 비교)

| 방식 | 실험이름 | logloss | accuracy | macro_f1 | 긴문장 logloss | 소요(s) |
|---|---|---|---|---|---|---|
| 나이브 베이즈 (NB 단계 최고) | mnb_tfidf_word12_a0.03_caliso | 0.4725 | 0.8326 | 0.8257 | 0.2863 | 67.8 |
| 단일 모델 (LR / SVM) | lr_tfidf_word12_C16 | 0.4603 | 0.8401 | 0.8326 | 0.2886 | 243.4 |
| 배깅 | bag10_lr_tfidf_word12_C16 | 0.4894 | 0.8334 | 0.8253 | 0.3286 | 843.4 |
| soft voting | vote_wopt_nb-lrw-lrc-svm | 0.4191 | 0.8622 | 0.8562 | 0.2598 | 0.4 |
| 스태킹 (부스팅 메타 포함) | stack_lgbm_nb-lrw-lrc-svm | 0.3806 | 0.8655 | 0.8601 | 0.2047 | 15.1 |

## 4-1. 기본 모델 (단일)

| 실험이름 | logloss | 긴문장 logloss | accuracy | macro_f1 | 소요(s) | 설정 | 비고 |
|---|---|---|---|---|---|---|---|
| lr_tfidf_word12_C16 | 0.4603 | 0.2886 | 0.8401 | 0.8326 | 243.4 | TF-IDF 단어 (1, 2), min_df=2, C=16 | 특징 168,581개(폴드 평균) |
| svm_tfidf_word12_C0.5_calsig | 0.4762 | 0.3117 | 0.8461 | 0.8393 | 220.8 | TF-IDF 단어 (1, 2), min_df=2, C=0.5, 보정 내부 5-fold | 특징 168,581개(폴드 평균) |
| lr_tfidf_word12_C4 | 0.4997 | 0.3504 | 0.8382 | 0.8305 | 210.3 | TF-IDF 단어 (1, 2), min_df=2, C=4 | 특징 168,581개(폴드 평균) |
| lr_tfidf_charwb15_C16 | 0.5396 | 0.3367 | 0.8069 | 0.7979 | 688.9 | TF-IDF 글자 char_wb (1, 5), min_df=2, max_features=200,000, C=16 | 특징 134,962개(폴드 평균) |
| svm_tfidf_word12_C0.1_calsig | 0.5420 | 0.3783 | 0.8250 | 0.8170 | 274.5 | TF-IDF 단어 (1, 2), min_df=2, C=0.1, 보정 내부 5-fold | 특징 168,581개(폴드 평균) |
| lr_tfidf_charwb15_C4 | 0.5441 | 0.3721 | 0.8114 | 0.8025 | 593.1 | TF-IDF 글자 char_wb (1, 5), min_df=2, max_features=200,000, C=4 | 특징 134,962개(폴드 평균) |
| lr_tfidf_word12_C1 | 0.6180 | 0.4919 | 0.8197 | 0.8101 | 173.9 | TF-IDF 단어 (1, 2), min_df=2, C=1 | 특징 168,581개(폴드 평균) |
| lr_tfidf_charwb15_C1 | 0.6340 | 0.4872 | 0.7990 | 0.7891 | 466.7 | TF-IDF 글자 char_wb (1, 5), min_df=2, max_features=200,000, C=1 | 특징 134,962개(폴드 평균) |

## 4-2. 배깅 vs 단일 모델

| 단일 모델 | 단일 logloss | 배깅 모델 | 배깅 logloss | 차이(배깅-단일) | 단일 acc | 배깅 acc | 소요(s) | 비고 |
|---|---|---|---|---|---|---|---|---|
| mnb_tfidf_word12_a0.03 | 0.4900 | bag10_mnb_tfidf_word12_a0.03 | 0.5082 | +0.0182 | 0.8291 | 0.8276 | 87.6 | 특징 168,581개(폴드 평균) |
| lr_tfidf_word12_C16 | 0.4603 | bag10_lr_tfidf_word12_C16 | 0.4894 | +0.0291 | 0.8401 | 0.8334 | 843.4 | 특징 168,581개(폴드 평균) |

## 4-3. soft voting

| 실험이름 | 사용 모델 | logloss | accuracy | macro_f1 | 설정 | 비고 |
|---|---|---|---|---|---|---|
| vote_wopt_nb-lrw-lrc-svm | nb=mnb_tfidf_word12_a0.03_caliso + lrw=lr_tfidf_word12_C16 + lrc=lr_tfidf_charwb15_C16 + svm=svm_tfidf_word12_C0.5_calsig | 0.4191 | 0.8622 | 0.8562 | 가중 평균 w=[NB 0.455, LR 단어 0.446, LR 글자 0.099, SVM 0.000] (OOF logloss 최소화, SLSQP, 합=1·0 이상) | 가중치를 OOF 전체로 찾음(약간 낙관적). 폴드별로 가중치를 다시 찾아 검증한 logloss 0.4192 |
| vote_wopt_nb-lrw | nb=mnb_tfidf_word12_a0.03_caliso + lrw=lr_tfidf_word12_C16 | 0.4207 | 0.8607 | 0.8545 | 가중 평균 w=[NB 0.479, LR 단어 0.521] (OOF logloss 최소화, SLSQP, 합=1·0 이상) | 가중치를 OOF 전체로 찾음(약간 낙관적). 폴드별로 가중치를 다시 찾아 검증한 logloss 0.4207 |
| vote_wopt_nb-lrw-svm | nb=mnb_tfidf_word12_a0.03_caliso + lrw=lr_tfidf_word12_C16 + svm=svm_tfidf_word12_C0.5_calsig | 0.4207 | 0.8607 | 0.8545 | 가중 평균 w=[NB 0.479, LR 단어 0.521, SVM 0.000] (OOF logloss 최소화, SLSQP, 합=1·0 이상) | 가중치를 OOF 전체로 찾음(약간 낙관적). 폴드별로 가중치를 다시 찾아 검증한 logloss 0.4207 |
| vote_mean_nb-lrw | nb=mnb_tfidf_word12_a0.03_caliso + lrw=lr_tfidf_word12_C16 | 0.4207 | 0.8605 | 0.8543 | 단순 평균 (가중치 동일) |  |
| vote_mean_nb-lrw-svm | nb=mnb_tfidf_word12_a0.03_caliso + lrw=lr_tfidf_word12_C16 + svm=svm_tfidf_word12_C0.5_calsig | 0.4276 | 0.8602 | 0.8540 | 단순 평균 (가중치 동일) |  |
| vote_mean_nb-lrw-lrc-svm | nb=mnb_tfidf_word12_a0.03_caliso + lrw=lr_tfidf_word12_C16 + lrc=lr_tfidf_charwb15_C16 + svm=svm_tfidf_word12_C0.5_calsig | 0.4283 | 0.8601 | 0.8539 | 단순 평균 (가중치 동일) |  |
| vote_wopt_nb-lrc | nb=mnb_tfidf_word12_a0.03_caliso + lrc=lr_tfidf_charwb15_C16 | 0.4374 | 0.8497 | 0.8433 | 가중 평균 w=[NB 0.659, LR 글자 0.341] (OOF logloss 최소화, SLSQP, 합=1·0 이상) | 가중치를 OOF 전체로 찾음(약간 낙관적). 폴드별로 가중치를 다시 찾아 검증한 logloss 0.4374 |
| vote_mean_nb-lrc | nb=mnb_tfidf_word12_a0.03_caliso + lrc=lr_tfidf_charwb15_C16 | 0.4416 | 0.8512 | 0.8445 | 단순 평균 (가중치 동일) |  |

## 4-4. 스태킹 (메타 모델: LightGBM 부스팅 / 로지스틱 회귀)

| 실험이름 | logloss | 긴문장 logloss | accuracy | macro_f1 | 소요(s) | 설정 | 비고 |
|---|---|---|---|---|---|---|---|
| stack_lgbm_nb-lrw-lrc-svm | 0.3806 | 0.2047 | 0.8655 | 0.8601 | 15.1 | LightGBM 메타 (num_leaves=15, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, early stopping 100라운드 · 학습 부분의 10%로 판단), 입력=기본 모델 log 확률 | 폴드별 best_iteration [119, 121, 169, 135, 126]; 1단계와 같은 폴드로 검증해 약간 낙관적일 수 있음 |
| stack_lr_nb-lrw-lrc-svm | 0.3850 | 0.2108 | 0.8653 | 0.8596 | 17.6 | LogisticRegression 메타 (C=1), 입력=기본 모델 log 확률 | 1단계와 같은 폴드로 검증해 약간 낙관적일 수 있음 |
