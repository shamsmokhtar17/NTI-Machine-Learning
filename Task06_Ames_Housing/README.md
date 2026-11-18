# Task 06 — Ames Housing (Regression)

Predict the sale price of a house.

- **Dataset:** `dataset/Ames_Housing_Dataset.csv` (1,460 rows, 80 features)
- **Target:** `SalePrice` (trained on `log(SalePrice)`)
- **Metrics:** MAE, MSE, RMSE, R² (log scale and real dollars)

## Steps
1. Data understanding and cleaning
   - `NA` = no feature → category `None`; `LotFrontage` filled **after the split** with the train median (no data leakage)
2. Outlier analysis (IQR): kept (real values); only the 2 houses with `GrLivArea ≥ 4000` (partial sales) removed; `log1p` for skewed features
3. EDA, encoding (ordinal quality, one-hot, target encoding from train only), train/test split (80/20), scaling fitted on train only
4. First models with GridSearchCV, then all models of the track + ElasticNet + Voting + Stacking
5. Model comparison with 5-fold cross-validation (R², RMSE)
6. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
7. Final evaluation on the test set: MAE, MSE, RMSE, R², Actual vs Predicted, residual plots, interpretation
8. Gradio app (last cell)

## Results
| Item | Value |
|---|---|
| Selected model | Ridge (`alpha=30`) |
| Best 3 after CV | Ridge, CatBoost, GradientBoosting |
| CV R² (5-fold) before → after tuning | Ridge 0.9062 → 0.9161; CatBoost 0.9022 → 0.9094; GradientBoosting 0.9009 → 0.9020 |
| Test MAE / MSE / RMSE (log scale) | 0.082 / 0.0144 / 0.120 |
| Test R² (log scale) | 0.909 |
| Test MAE / RMSE (real price) | $13,685 / $19,609 |
| Test R² (real price) | 0.927 |

**Interpretation:** the model explains 93% of the variation of the sale price; the median error is 5.9% of the price and 69% of the houses are predicted within 10%. Residuals are centred on 0 (mean 0.009). The top 3 models are within 0.014 CV R², and a regularized linear model (Ridge) beats the tree/boosting models after the log transformation and scaling, which means the relationship is close to linear in log scale. Lasso and ElasticNet with their default `alpha` shrink everything to zero (R² ≈ 0), so they were not selected.

**Gradio app:** the last cell of the notebook launches the app with the final model (`share=True` gives a temporary public link in Colab).

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
