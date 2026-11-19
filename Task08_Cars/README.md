# Task 08 — Cars Price Prediction (Regression)

Predict the price of a used car.

- **Dataset:** `dataset/cars_data.csv` (1,000 rows, 7 features)
- **Target:** `Price` (trained on `log(Price)`)
- **Metrics:** MAE, MSE, RMSE, R² (log scale and real dollars)

## Steps
1. Data understanding and cleaning
   - `Engine_Size` missing values filled **after the split** with the train median (the old code used the mean of the whole data = leakage)
2. Outlier analysis (IQR): the few very expensive cars (price ≥ 32,000, about 1%) removed; `log1p` of the target for the skew
3. EDA, encoding (one-hot Brand / Fuel_Type, label Transmission), train/test split (80/20), MinMax scaling fitted on train only
4. All models of the track + ElasticNet + Voting + Stacking
5. Model comparison with 5-fold cross-validation (R², RMSE)
6. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
7. Final evaluation on the test set: MAE, MSE, RMSE, R², Actual vs Predicted, residual plots, interpretation

## Results
| Item | Value |
|---|---|
| Selected model | CatBoost (`depth=4, iterations=600, learning_rate=0.03`) |
| Best 3 after CV | CatBoost, LightGBM, HistGradientBoosting |
| CV R² (5-fold) before → after tuning | CatBoost 0.9418 → 0.9506; LightGBM 0.9340 → 0.9419; HistGB 0.9328 → 0.9366 |
| Test MAE / MSE / RMSE (log scale) | 0.109 / 0.0175 / 0.132 |
| Test R² (log scale) | 0.954 |
| Test MAE / RMSE (real price) | $968 / $1,441 |
| Test R² (real price) | 0.935 |

**Interpretation:** the model explains 93.5% of the variation of the car price; the median error is 9.3% of the price (MAE about $968) and 52% of the cars are predicted within 10%. Residuals are centred on 0 (mean −0.005). Tuning improved all three models, mostly CatBoost (+0.009 CV R²). Boosting models clearly beat KNN, Decision Tree and AdaBoost, and linear models are already good (CV R² about 0.91). Lasso and ElasticNet with their default `alpha` shrink everything to zero (R² ≈ 0), so they were not selected.

**Limitations:** only 1,000 cars and 7 features (no model name, condition or accident history); the 11 most expensive cars (price ≥ 32,000) were removed, so the model is meant for typical cars below that price.

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
