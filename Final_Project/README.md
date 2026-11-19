# Final Project — Road Accident Severity Prediction (UK 2018)

## Problem
Predict whether a road accident is **Serious / Fatal** (≈20%) or **Slight** (≈80%) from the conditions of the accident: road, speed limit, weather, light, time, and the vehicles / drivers involved. This helps road authorities see which conditions are dangerous.

## Dataset
UK Department for Transport *Road Safety Data 2018*: Accidents (122,635), Vehicles (226,409) and Casualties (160,597), linked by `Accident_Index` (see `dataset/README.md`).

## What was done
1. **Data understanding & cleaning:** `-1` codes → missing, duplicates checked, time → hour / month, outliers checked with IQR (kept, they are real cases).
2. **EDA:** target balance, severity rate by speed limit / light / road type / area / hour / vehicle type, casualties table, correlations.
3. **Feature engineering:** vehicles summarised to one row per accident (motorcycle, pedal cycle, bus, heavy vehicle, driver ages, vehicle age).
4. **No data leakage:** `Casualty_Severity` and `Number_of_Casualties` are **not** used (they are the result of the accident, and the accident severity equals the worst casualty severity in 100% of the cases); the police-attendance column is dropped for the same reason. The split is stratified and made before any fitting; imputing, scaling and encoding are inside a `Pipeline` / `ColumnTransformer`.
5. **All models compared:** a Train-vs-Test table (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC for every model), then cross-validation on the train data (3-fold, PR-AUC). Models: Logistic Regression, SVM (linear), Naive Bayes, KNN, Decision Tree, Random Forest, ExtraTrees, Bagging, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
6. **GridSearchCV** on the top 3, then the final model is trained on all the train data; the **threshold** is chosen on out-of-fold train predictions (max F1).
7. **Final evaluation** on the test set (confusion matrix, ROC, PR, feature importance) and a **Streamlit app** (live deployment).

## Results
Best model (selected by 3-fold CV PR-AUC + `GridSearchCV` on the top 3): **CatBoost** (`depth=6, iterations=150, learning_rate=0.05`, class-balanced), decision threshold **0.51** (max F1 on out-of-fold train predictions).

| Test set (24,527 accidents) | Value |
|---|---|
| ROC-AUC | 0.709 |
| PR-AUC | 0.380 (random guess = 0.203) |
| Recall (Serious/Fatal) | 0.64 |
| Precision (Serious/Fatal) | 0.32 |
| F1 (Serious/Fatal) | 0.43 |
| Balanced accuracy | 0.647 |
| Accuracy | 0.652 (always "Slight" = 0.797) |

The model finds 3,166 of the 4,967 serious/fatal accidents (1,801 missed) and raises 6,726 false alarms.

**All models (test set, default threshold 0.5; models trained on 50,000 train rows)** - full table in `results/all_models_train_test.csv`:

| Model | Accuracy | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.647 | 0.651 | 0.428 | 0.711 | 0.386 |
| HistGradientBoosting | 0.640 | 0.661 | 0.426 | 0.710 | 0.384 |
| CatBoost | 0.649 | 0.649 | 0.428 | 0.710 | 0.383 |
| LightGBM | 0.654 | 0.644 | 0.430 | 0.709 | 0.382 |
| Random Forest | 0.696 | 0.528 | 0.413 | 0.699 | 0.373 |
| Gradient Boosting | 0.801 | 0.043 | 0.081 | 0.701 | 0.373 |
| ExtraTrees | 0.671 | 0.522 | 0.392 | 0.672 | 0.356 |
| Logistic Regression | 0.639 | 0.609 | 0.406 | 0.680 | 0.354 |
| SVM (linear) | 0.640 | 0.603 | 0.404 | 0.679 | 0.353 |
| AdaBoost | 0.799 | 0.035 | 0.066 | 0.690 | 0.350 |
| Decision Tree | 0.636 | 0.625 | 0.410 | 0.678 | 0.342 |
| KNN | 0.801 | 0.066 | 0.119 | 0.662 | 0.328 |
| Bagging | 0.783 | 0.145 | 0.213 | 0.653 | 0.313 |
| Naive Bayes | 0.663 | 0.420 | 0.336 | 0.603 | 0.263 |

**Cross-validation + GridSearchCV (top 3, PR-AUC):** CatBoost 0.3574 -> 0.3580, Gradient Boosting 0.3562 -> 0.3569, Random Forest 0.3556 -> 0.3556 (`results/gridsearch_top3.csv`). The top 3 are within 0.002, so tuning changed almost nothing.

**Interpretation**
- The model ranks serious accidents clearly better than chance (PR-AUC 0.38 vs 0.20), but the problem is hard: the data only has the conditions of the accident, not the speed of the vehicles, seat belts or the angle of impact. Precision is low (32%), so it is a risk-ranking tool, not a decision tool.
- Accuracy is not a good measure here: models with a high accuracy (~0.80: KNN, AdaBoost, Gradient Boosting) just say "Slight" almost always (recall 4-7%). Bagging overfits (Train F1 0.98 vs Test 0.21).
- Most important features (permutation importance): motorcycle involved, number of vehicles, pedal cycle involved, speed limit, police force area, urban/rural.
- Casualty information (`Casualty_Severity`, `Number_of_Casualties`) was **not** used: accident severity equals the worst casualty severity in 100% of the accidents, so it would be data leakage.

**Recommendations:** focus safety measures (speed control, rider protection, lighting) on high-speed rural roads and on roads with many motorcycles / cyclists; use the model to rank conditions by risk, not to judge individual drivers.

**Limitations:** one year (2018) and one country; no exact speed / seat belt / impact data; modest precision.

## Structure
```
Final_Project/
  dataset/      the 3 csv files
  notebooks/    final_project.ipynb
  models/       best_model.joblib (pipeline + threshold)
  results/      plots, comparison table, GridSearchCV table, final metrics
  app.py              Gradio app
  streamlit_app.py    Streamlit app (used for the live deployment)
  requirements.txt
  README.md
```

## How to run
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
(`app.py` is the same app with Gradio: `pip install gradio` then `python app.py`.)

**Live demo (Streamlit Community Cloud):** https://nti-machine-learning-yahbpmz7kaupiaw6nfkate.streamlit.app/
