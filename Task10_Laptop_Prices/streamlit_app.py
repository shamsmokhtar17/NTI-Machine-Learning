"""Laptop Price Prediction - Streamlit app (Task 10 deployment).
Loads the trained model + the same preprocessing objects used in the notebook."""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_saved():
    return joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "laptop_model.joblib"))


saved = load_saved()
model, imputer_na, scaler = saved["model"], saved["imputer_na"], saved["scaler"]
na_cols, skewed, num_cols = saved["na_cols"], saved["skewed"], saved["num_cols"]
onehot_cols, high_card, target_means = saved["onehot_cols"], saved["high_card"], saved["target_means"]
X_columns, y_train_mean, typical_laptop, choices = saved["X_columns"], saved["y_train_mean"], saved["typical_laptop"], saved["choices"]


def prepare_row(row):
    row = row.copy()
    for col in high_card:
        row[col] = row[col].map(target_means[col]).fillna(y_train_mean)      # target encoding learned on train
    row = pd.get_dummies(row, columns=onehot_cols)
    row = row.reindex(columns=X_columns, fill_value=0).astype(float)         # same columns / order as training
    row[na_cols] = imputer_na.transform(row[na_cols])                        # same fill values as the train data
    row[skewed] = np.log1p(row[skewed])
    row[num_cols] = scaler.transform(row[num_cols])                          # same scaler as the train data
    return row


def predict_price(company, type_name, ram, inches, weight, resolution, touchscreen, ips,
                  cpu_brand_, cpu_ghz, ssd, hdd, gpu_brand_, os_name):
    x_res, y_res = (int(v) for v in resolution.split("x"))
    row = typical_laptop.copy()          # the other columns keep the values of a typical laptop
    row["Company"] = company
    row["TypeName"] = type_name
    row["Ram"] = int(ram)
    row["Inches"] = inches
    row["Weight"] = weight
    row["X_res"] = x_res
    row["Y_res"] = y_res
    row["PPI"] = np.sqrt(x_res ** 2 + y_res ** 2) / inches
    row["Touchscreen"] = int(touchscreen)
    row["IPS"] = int(ips)
    row["Cpu_brand"] = cpu_brand_
    row["Cpu_GHz"] = cpu_ghz
    row["SSD"] = ssd
    row["HDD"] = hdd
    row["Hybrid"] = 0
    row["Flash"] = 0
    row["Gpu_brand"] = gpu_brand_
    row["OS"] = os_name
    price = np.expm1(model.predict(prepare_row(row)))[0]
    return price




st.set_page_config(page_title="Laptop Price Prediction", page_icon="💻")
st.title("Laptop Price Prediction App")
st.write("Enter the main laptop specs to estimate its price. Other details are set to a typical laptop.")

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Company", choices["Company"], index=choices["Company"].index("Dell") if "Dell" in choices["Company"] else 0)
    type_name = st.selectbox("Type", choices["TypeName"], index=choices["TypeName"].index("Notebook") if "Notebook" in choices["TypeName"] else 0)
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    resolution = st.selectbox("Screen Resolution", ["1366x768", "1600x900", "1920x1080", "2560x1440", "2560x1600", "2880x1800", "3200x1800", "3840x2160"], index=2)
    touchscreen = st.checkbox("Touchscreen", value=False)
    ips = st.checkbox("IPS Panel", value=False)
with col2:
    cpu_brand_ = st.selectbox("CPU", choices["Cpu_brand"], index=choices["Cpu_brand"].index("Intel Core i5") if "Intel Core i5" in choices["Cpu_brand"] else 0)
    cpu_ghz = st.number_input("CPU Speed (GHz)", min_value=0.9, max_value=4.0, value=2.5, step=0.1)
    ssd = st.selectbox("SSD (GB)", [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], index=6)
    hdd = st.selectbox("HDD (GB)", [0, 500, 1024, 2048], index=0)
    gpu_brand_ = st.selectbox("GPU Brand", choices["Gpu_brand"], index=choices["Gpu_brand"].index("Intel") if "Intel" in choices["Gpu_brand"] else 0)
    os_name = st.selectbox("Operating System", choices["OS"], index=choices["OS"].index("Windows") if "Windows" in choices["OS"] else 0)

if st.button("Predict price", type="primary"):
    price = predict_price(company, type_name, ram, inches, weight, resolution, touchscreen, ips,
                          cpu_brand_, cpu_ghz, ssd, hdd, gpu_brand_, os_name)
    st.success(f"Estimated Price: {price:,.0f}")
st.caption("Trained CatBoost model on log(price). Prices are in the units of the dataset. For learning only.")
