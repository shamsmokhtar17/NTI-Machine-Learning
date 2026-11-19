# Task 09 — Diabetes Patients (Classification)

Predict whether a patient is diabetic.

- **Dataset:** `dataset/diabetes_patients_10000_full.csv` (10,000 rows, 8 features)
- **Target:** `Class` (1 = Diabetic ≈ 91%, 0 = Non-Diabetic ≈ 9%) — imbalanced, with the minority being the *healthy* class
- **Metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix

## Steps
1. Data understanding and cleaning (duplicates, `Patient_ID` dropped, missing values filled **after the split** with train statistics — no data leakage)
2. Outlier analysis (IQR): nothing to remove; scaling fitted on train only
3. EDA, encoding, stratified train/test split (80/20)
4. Imbalance: SMOTE, random under-sampling, class weights (Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree)
5. All models of the track + Voting + Stacking (confusion matrix, ROC and PR curves)
6. Model comparison with 5-fold stratified cross-validation (ROC-AUC, macro F1, balanced accuracy)
7. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
8. Decision threshold chosen on training data (Youden's J); final evaluation on the test set with interpretation
9. Check without `HbA1c`; Gradio app (last cell)

## Results
| Item | Value |
|---|---|
| Selected model | LightGBM (`learning_rate=0.1, n_estimators=300, num_leaves=15`) |
| Best 3 after CV | LightGBM, Hist Gradient Boosting, Gradient Boosting |
| CV ROC-AUC (5-fold) before → after tuning | 0.999994 → 0.999995 (LightGBM); the other two unchanged (0.99999) |
| Decision threshold (chosen on train, Youden's J) | 0.973 |
| Test Accuracy / Balanced Accuracy | 0.9995 / 0.9971 |
| Test Recall (Diabetic / Non-Diabetic) | 100% / 99.4% (only 1 error in 2,000 patients) |
| Test Precision (Diabetic / Non-Diabetic) | 99.9% / 100% |
| Test ROC-AUC | 1.000 |
| ROC-AUC **without HbA1c** | 0.878 |

**Interpretation:** the model makes 1 error in 2,000 test patients. The result is almost perfect because the problem is very easy in this dataset: `HbA1c` is the test that defines diabetes (≥ 6.5%) and all Non-Diabetic patients have HbA1c ≤ 6.0, so one column almost decides the answer. Without `HbA1c` the ROC-AUC drops to 0.88. Accuracy must also be compared with the 91.5% of a model that always says "Diabetic", so Balanced Accuracy and the recall of the minority class (Non-Diabetic) are the numbers to look at. Tuning changed almost nothing because the top boosting models are already at the ceiling.

**Limitations:** the data looks synthetic and very clean, so these scores will not transfer to real patients; this is a learning project, not a medical tool.

**Gradio app:** the last cell of the notebook launches the app with the final model and threshold (`share=True` gives a temporary public link in Colab).

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
