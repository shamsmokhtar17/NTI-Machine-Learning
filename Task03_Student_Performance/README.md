# Task 03 — Student Performance Prediction

Regression: predict the **Performance Index** (10–100) of a student.

- **Dataset:** `dataset/Student_Performance.csv` (10,000 rows, 5 features)
- **Target:** `Performance Index`
- **Metrics:** MAE, MSE, RMSE, R²

## Steps
1. Data understanding and cleaning (duplicates, missing values)
   - `Extracurricular Activities` missing (15%) → own category `Unknown` (no data leakage)
   - `Sample Question Papers Practiced` dropped: 28% missing and almost no correlation with the target (≈ 0.04)
   - the unnamed first column is renamed `Hours Studied`
2. Outlier analysis (IQR): no extreme values → nothing removed, no target transformation needed
3. EDA, encoding, train/test split (80/20), scaling (fit on train only)
4. Linear Regression baseline
5. All models of the track (Linear, Lasso, Ridge, ElasticNet, KNN, LinearSVR, Decision Tree, Random Forest, ExtraTrees, Bagging, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, CatBoost, LightGBM, Voting, Stacking)
6. Model comparison with 5-fold cross-validation (R², RMSE)
7. `GridSearchCV` on the best 3 single models (`best_params_`, `best_score_`, before/after table)
8. Final evaluation on the test set: MAE, MSE, RMSE, R², Actual vs Predicted, residual plots, interpretation

## Results
| Item | Value |
|---|---|
| Selected model | Linear Regression (`fit_intercept=True`) |
| Best 3 after CV | Ridge (`alpha=10`), Linear Regression, LinearSVR (`C=1, epsilon=0.1`) |
| CV R² (5-fold) before → after tuning | 0.9877 → 0.9877 (Ridge / Linear Regression), 0.9875 → 0.9877 (LinearSVR) |
| Test MAE | 1.70 |
| Test MSE | 4.51 |
| Test RMSE | 2.12 |
| Test R² | 0.988 |

**Interpretation:** the model explains 98.8% of the variation of the Performance Index and is on average 1.7 points away from the real value (98% of the students are within ±5 points). The top models differ by less than 0.0005 in CV R², so the data is almost linear and a simple model is enough; tuning gives no real gain. Residuals are centred on 0 (mean 0.06).

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
