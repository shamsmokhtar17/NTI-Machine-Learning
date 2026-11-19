# Task 07 — Apartments Prices (Regression)

Predict the price (EGP) of an apartment.

- **Dataset:** `dataset/Apartments_Prices_Dataset.csv` (5,578 rows, 10 columns)
- **Target:** `price` (trained on `log(price)`)
- **Metrics:** MAE, MSE, RMSE, R² (log scale and real EGP)

## Steps
1. Data understanding and cleaning
   - text → numbers (remove `EGP`, `m²`, `rooms`...), duplicates removed, shifted `seller_type` values repaired
   - listings with an absurd price per m² removed; impossible values set to missing and filled **after the split** (train median, no data leakage)
   - `price_per_sqm` dropped (computed from the target)
2. Outlier analysis (IQR): real luxury apartments kept; `log1p` for skewed features and scaling fitted on train only
3. EDA, encoding, train/test split (80/20)
4. All models of the track + ElasticNet + Voting + Stacking
5. Model comparison with 5-fold cross-validation (R², RMSE)
6. `GridSearchCV` on the best 3 models (`best_params_`, `best_score_`, before/after table)
7. Final evaluation on the test set: MAE, MSE, RMSE, R², Actual vs Predicted, residual plots, interpretation
8. Gradio app (last cell)

## Results
| Item | Value |
|---|---|
| Selected model | HistGradientBoosting (`learning_rate=0.03, max_iter=400, max_depth=None`) |
| Best 3 after CV | LightGBM, HistGradientBoosting, XGBoost |
| CV R² (5-fold) before → after tuning | LightGBM 0.6048 → 0.6055; HistGB 0.6041 → 0.6058; XGBoost 0.5960 → 0.5801 |
| Test MAE / MSE / RMSE (log scale) | 0.269 / 0.125 / 0.353 |
| Test R² (log scale) | 0.663 |
| Test MAE / RMSE (real price) | 1,726,507 / 2,937,080 EGP |
| Test R² (real price) | 0.510 |

**Interpretation:** the model explains 66% of the variation of log(price) and 51% of the real price; the median error is about 21% of the price and 26% of the apartments are predicted within 10%. Residuals are centred on 0 (mean 0.005). The top 3 boosting models are within 0.01 CV R² of each other, so tuning helped very little. XGBoost got a lower score after tuning because its default settings (deeper trees) were not in the small search grid (depth 2–3).

**Limitations:** the dataset has no location (city / compound), which is the strongest driver of an apartment price, and the prices are asking prices; this limits the accuracy. Lasso and ElasticNet with their default `alpha` shrink everything to zero (R² ≈ 0).

**Gradio app:** the last cell of the notebook launches the app with the final model (`share=True` gives a temporary public link in Colab).

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
