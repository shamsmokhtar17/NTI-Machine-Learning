# Task 01 — Stroke Prediction

Binary classification: predict whether a patient will have a stroke.

- **Dataset:** `dataset/healthcare-dataset-stroke-data.csv` (Kaggle Stroke Prediction Dataset, 5,110 rows)
- **Target:** `stroke` (~5% positives, imbalanced → Accuracy is misleading; we use Recall, F1, ROC-AUC and PR-AUC)

## Steps
1. Data understanding, cleaning (duplicates, missing `bmi`), outlier analysis (kept: real medical values)
2. EDA and visualization
3. Encoding, stratified train/test split (80/20), scaling and imputation fitted on the train set only
4. Logistic Regression with imbalance handling (SMOTE, under sampling, class weights)
5. All models of the track: confusion matrix, classification report, ROC and PR curves
6. Model comparison with 5-fold cross-validation (PR-AUC)
7. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
8. Final `Pipeline` + `ColumnTransformer`, decision threshold chosen on training data, final evaluation on the test set
9. Model saved and deployed with Gradio (`app.py`)

## Results
Best model: **Gradient Boosting** (selected with 5-fold CV PR-AUC, then tuned with GridSearchCV).

| Item | Value |
|---|---|
| Best model | Gradient Boosting |
| best_params_ | learning_rate=0.05, max_depth=3, n_estimators=100 |
| CV PR-AUC before → after tuning | 0.203 → 0.216 |
| Decision threshold (chosen on train, max F2) | 0.07 |
| Test ROC-AUC / PR-AUC | 0.836 / 0.214 (random guess PR-AUC = 0.049) |
| Test Recall / Precision / F1 (stroke) | 0.78 / 0.17 / 0.28 |

The model finds 39 of the 50 stroke patients in the test set (11 missed) and raises 189 false alarms. Precision is low because stroke is rare (~5%); for screening, catching real cases matters more.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb      # Run All -> creates model/stroke_model.joblib
python app.py                        # Gradio app on http://127.0.0.1:7860
```

## Deployment
`app.py` loads the trained pipeline from `model/stroke_model.joblib`, takes the patient data and shows the predicted probability and risk level.
Live demo (Hugging Face Spaces): _<add link>_

> Educational project, not a medical diagnosis tool.
