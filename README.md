# NTI Machine Learning for Data Analysis

All course tasks (Shams Mokhtar Mohamed).

Every task follows the same steps: data understanding & cleaning → outlier analysis → EDA → split first, then imputation / encoding / scaling fitted on the **train data only** (no leakage) → model comparison with cross-validation → `GridSearchCV` on the top 3 → final evaluation on the test set with interpretation.

## Results

| Folder | Dataset | Type | Best model | Main test result |
|---|---|---|---|---|
| [Task01_Stroke_Dataset](Task01_Stroke_Dataset/) | Stroke Prediction | Classification | Gradient Boosting | ROC-AUC 0.836, PR-AUC 0.214, Recall 0.78 |
| [Task02_House_Rent](Task02_House_Rent/) | House Rent | Regression | LightGBM | R² (log) 0.793 |
| [Task03_Student_Performance](Task03_Student_Performance/) | Student Performance | Regression | Linear Regression | R² 0.988, MAE 1.70 |
| [Task04_Adult_Census_Income](Task04_Adult_Census_Income/) | Adult Census Income | Classification | CatBoost | ROC-AUC 0.923, F1 0.72 |
| [Task05_Bank_Marketing](Task05_Bank_Marketing/) | Bank Marketing | Classification | CatBoost | ROC-AUC 0.805, F1 0.73 |
| [Task06_Ames_Housing](Task06_Ames_Housing/) | Ames Housing | Regression | Ridge | R² (real price) 0.927 |
| [Task07_Apartments_Prices](Task07_Apartments_Prices/) | Apartments Prices | Regression | HistGradientBoosting | R² (log) 0.663 |
| [Task08_Cars](Task08_Cars/) | Cars Data | Regression | CatBoost | R² (real price) 0.935 |
| [Task09_Diabetes_Patients](Task09_Diabetes_Patients/) | Diabetes Patients | Classification | LightGBM | ROC-AUC 1.000 (0.878 without HbA1c) |
| [Task10_Laptop_Prices](Task10_Laptop_Prices/) | Laptop Prices | Regression | CatBoost | R² (real price) 0.783, (log) 0.890 |
| [Final_Project](Final_Project/) | UK Road Safety 2018 (3 tables) | Classification | CatBoost | ROC-AUC 0.709, PR-AUC 0.380, Recall 0.64 |

Each folder has its own `README.md` (steps, full results, interpretation, limitations), the `notebook.ipynb` and the `dataset/`. Gradio apps are included in Task01 (`app.py`), Task05, Task06, Task07, Task09 and Task10.

## Live deployments
- **Final Project:** [Road Accident Severity app (Streamlit)](https://nti-machine-learning-yahbpmz7kaupiaw6nfkate.streamlit.app/)
- **Task deployment (Task10 - Laptop Prices):** _link will be added_

## How to run
```
pip install -r requirements.txt
jupyter notebook
```
Or open any `notebook.ipynb` in Google Colab (the dataset path is `dataset/<file>.csv`; if you upload the CSV next to the notebook it also works).
