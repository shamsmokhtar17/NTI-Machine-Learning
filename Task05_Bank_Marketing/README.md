# Task 05 — Bank Marketing (Classification)

Predict whether a client subscribes to a term deposit.

- **Dataset:** `dataset/bank.csv` (8,912 rows, 16 features after removing `duration`)
- **Target:** `deposit` (yes ≈ 47%, almost balanced)
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix

## Steps
1. Data understanding and cleaning
   - `duration` removed: it is only known after the call (data leakage)
   - missing categorical values → `unknown`; missing numeric values → median of the train data (after the split)
2. Outlier analysis (IQR): kept (real values), scaling fitted on train only
3. EDA, encoding, stratified train/test split (80/20)
4. All models of the track + Voting + Stacking (confusion matrix, ROC and PR curves)
5. Model comparison with 5-fold stratified cross-validation (ROC-AUC, F1, Accuracy)
6. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
7. Decision threshold chosen on training data; final evaluation on the test set with interpretation
8. Gradio app using the final model

## Results
| Item | Value |
|---|---|
| Selected model | CatBoost (`depth=6, iterations=600, learning_rate=0.03`) |
| Best 3 after CV | CatBoost, Hist Gradient Boosting, LightGBM |
| CV ROC-AUC (5-fold) before → after tuning | CatBoost 0.8197 → 0.8199; HistGB 0.8125 → 0.8146; LightGBM 0.8092 → 0.8145 |
| Decision threshold (chosen on train, max F1) | 0.39 |
| Test Accuracy | 0.731 |
| Test Precision / Recall / F1 (Deposit) | 0.70 / 0.76 / 0.73 |
| Test ROC-AUC | 0.805 |
| Test PR-AUC | 0.792 (random = 0.48) |

**Interpretation:** the model finds 647 of the 846 clients who subscribe (recall 76%) and 70% of the clients it flags really subscribe. Accuracy (73%) is 20 points above the "most frequent class" baseline (53%). The scores are lower than in notebooks that keep `duration`, on purpose: `duration` (call length) is only known after the call, so using it would be data leakage. The top 3 boosting models are within 0.005 ROC-AUC of each other, and tuning improved them only slightly.

**Gradio app:** the last cell of the notebook launches the app with the final model (`share=True` gives a temporary public link in Colab).

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
