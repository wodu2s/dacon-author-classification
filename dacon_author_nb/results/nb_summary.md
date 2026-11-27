# 나이브 베이즈 실험 요약 (5-fold OOF 기준)

- 실험 수: 40개  |  주 지표: OOF logloss (낮을수록 좋음)
- 긴문장 logloss: 테스트 최소 길이 이상인 학습 행만으로 계산한 OOF logloss (테스트 조건 근사, 참고용)

## 1단계: 모델 × 벡터라이저 × 단위 (alpha=1)

| 실험이름 | 모델 | 벡터라이저 | 단위 | n-gram | logloss | accuracy | macro_f1 | n_features | 소요(s) |
|---|---|---|---|---|---|---|---|---|---|
| mnb_tfidf_word12_a1 | MultinomialNB | TfidfVectorizer | word | (1,2) | 0.8764 | 0.6751 | 0.6046 | 168581 | 19.0 |
| mnb_count_word11_a1 | MultinomialNB | CountVectorizer | word | (1,1) | 0.8869 | 0.7564 | 0.7471 | 21473 | 6.6 |
| mnb_tfidf_charwb24_a1 | MultinomialNB | TfidfVectorizer | char_wb | (2,4) | 0.8891 | 0.6677 | 0.6044 | 55281 | 52.3 |
| mnb_tfidf_word11_a1 | MultinomialNB | TfidfVectorizer | word | (1,1) | 0.9089 | 0.7075 | 0.6730 | 21473 | 6.8 |
| mnb_tfidf_charwb15_a1 | MultinomialNB | TfidfVectorizer | char_wb | (1,5) | 1.0462 | 0.6307 | 0.5453 | 134962 | 81.8 |
| mnb_count_word12_a1 | MultinomialNB | CountVectorizer | word | (1,2) | 1.2963 | 0.8051 | 0.7976 | 168581 | 18.7 |
| bnb_count_word11_a1 | BernoulliNB | CountVectorizer | word | (1,1) | 1.7213 | 0.6890 | 0.6612 | 21473 | 6.6 |
| mnb_count_charwb24_a1 | MultinomialNB | CountVectorizer | char_wb | (2,4) | 3.4925 | 0.7300 | 0.7195 | 55281 | 49.8 |
| bnb_count_word12_a1 | BernoulliNB | CountVectorizer | word | (1,2) | 3.5981 | 0.7195 | 0.6793 | 168581 | 19.0 |
| mnb_count_charwb15_a1 | MultinomialNB | CountVectorizer | char_wb | (1,5) | 4.0996 | 0.7550 | 0.7453 | 134962 | 79.0 |
| bnb_count_charwb15_a1 | BernoulliNB | CountVectorizer | char_wb | (1,5) | 9.0379 | 0.6339 | 0.6064 | 134962 | 81.0 |
| bnb_count_charwb24_a1 | BernoulliNB | CountVectorizer | char_wb | (2,4) | 9.0872 | 0.5940 | 0.5661 | 55281 | 51.2 |

## 2단계: alpha(라플라스 스무딩) 탐색 — OOF logloss

| 설정 | α=0.01 | α=0.03 | α=0.1 | α=0.3 | α=1 |
|---|---|---|---|---|---|
| MNB · TF-IDF · word (1,2) | 0.4949 | 0.4900 | 0.5256 | 0.6341 | 0.8764 |
| MNB · TF-IDF · char_wb (2,4) | 0.6618 | 0.6676 | 0.6868 | 0.7283 | 0.8891 |
| MNB · Count · word (1,1) | 0.9688 | 0.9178 | 0.8792 | 0.8662 | 0.8869 |
| BNB · Count · word (1,1) | 1.3896 | 1.3728 | 1.3941 | 1.4802 | 1.7213 |

## 3단계: 전처리 변형 (keep=유지 / nostop=불용어 제거 / nopunct=문장부호 제거) — OOF logloss

| 설정 | keep | nostop | nopunct | nostopnopunct |
|---|---|---|---|---|
| mnb_tfidf_charwb24_a0.01 | 0.6618 | 0.6831 | 0.7999 | 0.8278 |
| mnb_tfidf_word12_a0.03 | 0.4900 | 0.5727 | 0.5884 | 0.6905 |

## 4단계: 확률 보정 (CalibratedClassifierCV)

| 실험이름 | 보정 | logloss | 긴문장 logloss | accuracy | macro_f1 | 소요(s) |
|---|---|---|---|---|---|---|
| mnb_tfidf_word12_a0.03 | none | 0.4900 | 0.3118 | 0.8291 | 0.8219 | 18.9 |
| mnb_tfidf_word12_a0.03_calsig | sigmoid | 0.5063 | 0.3228 | 0.8317 | 0.8245 | 68.5 |
| mnb_tfidf_word12_a0.03_caliso | isotonic | 0.4725 | 0.2863 | 0.8326 | 0.8257 | 67.8 |
| mnb_count_word12_a1 | none | 1.2963 | 1.1908 | 0.8051 | 0.7976 | 18.7 |
| mnb_count_word12_a1_calsig | sigmoid | 0.6824 | 0.4958 | 0.8069 | 0.7993 | 66.8 |
| mnb_count_word12_a1_caliso | isotonic | 0.5519 | 0.3668 | 0.8117 | 0.8041 | 65.9 |
| bnb_count_word11_a0.03 | none | 1.3728 | 1.1646 | 0.7325 | 0.7130 | 6.7 |
| bnb_count_word11_a0.03_calsig | sigmoid | 0.8355 | 0.6481 | 0.7334 | 0.7140 | 23.4 |
| bnb_count_word11_a0.03_caliso | isotonic | 0.7121 | 0.5512 | 0.7398 | 0.7229 | 23.2 |

## OOF logloss 상위 5개

| 실험이름 | 모델 | 벡터라이저 | 단위 | n-gram | alpha | 보정 | 전처리 | logloss | 긴문장 logloss | accuracy | macro_f1 | 소요(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnb_tfidf_word12_a0.03_caliso | MultinomialNB | TfidfVectorizer | word | (1,2) | 0.03 | isotonic | keep | 0.4725 | 0.2863 | 0.8326 | 0.8257 | 67.8 |
| mnb_tfidf_word12_a0.03 | MultinomialNB | TfidfVectorizer | word | (1,2) | 0.03 | none | keep | 0.4900 | 0.3118 | 0.8291 | 0.8219 | 18.9 |
| mnb_tfidf_word12_a0.01 | MultinomialNB | TfidfVectorizer | word | (1,2) | 0.01 | none | keep | 0.4949 | 0.3050 | 0.8251 | 0.8177 | 18.9 |
| mnb_tfidf_word12_a0.03_calsig | MultinomialNB | TfidfVectorizer | word | (1,2) | 0.03 | sigmoid | keep | 0.5063 | 0.3228 | 0.8317 | 0.8245 | 68.5 |
| mnb_tfidf_word12_a0.1 | MultinomialNB | TfidfVectorizer | word | (1,2) | 0.1 | none | keep | 0.5256 | 0.3562 | 0.8254 | 0.8168 | 18.9 |
