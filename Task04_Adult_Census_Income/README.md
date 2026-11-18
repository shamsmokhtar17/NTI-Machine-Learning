# Task 04 — Adult Census Income (Classification)

Predict whether a person earns more than 50K a year.

- **Dataset:** `dataset/Adult_Census_Income.csv` (32,561 rows, 14 features)
- **Target:** `income` (1 = `>50K`, about 24% of the people)
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix

## Steps
1. Data understanding and cleaning (duplicates, `?` → category `Unknown`, no data leakage)
2. Outlier analysis (IQR): kept (real values), `log1p` for capital gain/loss, scaling
3. EDA, encoding, stratified train/test split (80/20), target encoding and scaling fitted on train only
4. Imbalance: SMOTE, random under-sampling, class weights (Logistic Regression, KNN, Naive Bayes)
5. All models of the track + Voting + Stacking (confusion matrix, ROC and PR curves)
6. Model comparison with 5-fold stratified cross-validation (PR-AUC)
7. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
8. Decision threshold chosen on training data; final evaluation on the test set with interpretation

## Results
| Item | Value |
|---|---|
| Selected model | CatBoost (`depth=4, iterations=600, learning_rate=0.1`) |
| Best 3 after CV | CatBoost, LightGBM, Hist Gradient Boosting |
| CV PR-AUC (5-fold) before → after tuning | CatBoost 0.8315 → 0.8321; LightGBM 0.8311 → 0.8311; HistGB 0.8291 → 0.8291 |
| Decision threshold (chosen on train, max F1) | 0.37 |
| Test Accuracy | 0.862 |
| Test Precision / Recall / F1 (>50K) | 0.70 / 0.75 / 0.72 |
| Test ROC-AUC | 0.923 |
| Test PR-AUC | 0.822 (random = 0.24) |

**Interpretation:** the model finds 1,174 of the 1,568 people who earn >50K (recall 75%) and 70% of the people it flags are right. Accuracy (86%) is only 10 points above the "always <=50K" baseline (76%), so Recall, F1 and PR-AUC are the more honest numbers. The top boosting models are almost identical (CV PR-AUC within 0.003), so tuning gave almost no extra gain. The data is a 1994 US census and includes `sex` and `race`, so it should not be used to judge real people.

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
