# Laptop Prices

- **Dataset:** Laptop Prices (`dataset/laptopData.csv`, 1,303 rows -> 1,244 after cleaning)
- **Task type:** Regression
- **Target:** `Price` (trained on `log(Price)`)

## Steps
1. Data understanding & cleaning (duplicates, empty rows, `?`, wrong screen sizes)
2. Feature engineering from text columns (resolution/PPI, CPU, storage, GPU, OS)
3. Outliers (IQR) - kept, handled with log transform + scaling
4. EDA
5. Split, then fill missing values / encode `Company` / scale using train data only (no leakage)
6. Models comparison, CV, GridSearchCV on the top 3, final evaluation, Gradio app

## Results
Best model (GridSearchCV on the top 3 by CV): **CatBoost** (`depth=6, iterations=600, learning_rate=0.1`), CV R² = 0.891.

| Metric (test set) | log(Price) | Real price |
|---|---|---|
| MAE | 0.146 | 9,087 |
| MSE | 0.0386 | 2.53e8 |
| RMSE | 0.197 | 15,915 |
| R² | 0.890 | 0.783 |

- Median error ≈ 11% of the real price; 46% of the laptops are predicted within 10%.
- Tuning: XGBoost 0.881 → 0.884, CatBoost 0.877 → 0.891, HistGradientBoosting 0.871 → 0.876 (CV R²).
- Caveats: only 1,244 laptops; exact CPU/GPU model reduced to brand; prices are in the dataset's units (old listing), not current prices.

## How to run
```
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
