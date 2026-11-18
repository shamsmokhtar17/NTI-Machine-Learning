# Task 02 — House Rent Prediction

Regression: predict the monthly rent of a house / apartment.

- **Dataset:** `dataset/House_Rent_Dataset.csv` (4,746 rows, 6 Indian cities)
- **Target:** `Rent` (very skewed: a few luxury houses cost millions → trained on `log(Rent)`)
- **Metrics:** MAE, MSE, RMSE, R²

## Steps
1. Data understanding and cleaning (duplicates, missing values)
   - the target `Rent` is never filled (5 rows removed); `Area Type` missing → `Unknown`; `Size` / `Bathroom` filled with the train median **after** the split (no data leakage)
2. Outlier analysis (IQR): extreme rents are real luxury houses → kept, handled with the log transformation of the target
3. EDA, encoding, train/test split (80/20), scaling
4. Linear Regression baseline, then log transformation of the target
5. All models of the track (Linear, Lasso, Ridge, ElasticNet, KNN, SVM, Decision Tree, Random Forest, ExtraTrees, Bagging, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, CatBoost, LightGBM, Voting, Stacking)
6. Model comparison with 5-fold cross-validation (R², RMSE)
7. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
8. Final evaluation on the test set: MAE, MSE, RMSE, R² (log and real scale), Actual vs Predicted, residual plots, interpretation

## Results
Best model: **LightGBM** (top 3 by 5-fold CV R² were CatBoost, GradientBoosting and LightGBM; after `GridSearchCV` LightGBM had the best `best_score_`).

| Item | Value |
|---|---|
| Best model | LightGBM |
| best_params_ | learning_rate=0.03, n_estimators=200, num_leaves=15 |
| CV R² before → after tuning | 0.812 → 0.817 |
| Test R² (log scale) | 0.793 |
| Test MAE / RMSE (real rent) | 15,798 / 118,296 |
| Median absolute error (real rent) | 3,687 (≈ 23% of the rent) |
| Test R² (real rent) | 0.21 (0.75 without the 1% most expensive houses) |

The R² on the real rent scale is very sensitive to the few luxury houses (3.5M, 1.2M) in the test set; the log-scale R² is the more reliable number. Linear models (Ridge / Linear Regression) already reach CV R² ≈ 0.80 on the log target, so the gain from boosting is small; Lasso / ElasticNet with default regularization perform poorly (CV R² ≈ 0.32).

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
