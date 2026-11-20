"""Laptop Price Prediction - Gradio app (Task 10 deployment).
Loads the trained model + the same preprocessing objects used in the notebook."""
import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

saved = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "laptop_model.joblib"))
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
    if any(v is None for v in [inches, weight, cpu_ghz, ssd, hdd]):
        return " "
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
    return f"Estimated Price: {price:,.0f}"


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(choices["Company"], label="Company", value="Dell"),
        gr.Dropdown(choices["TypeName"], label="Type", value="Notebook"),
        gr.Dropdown([2, 4, 6, 8, 12, 16, 24, 32, 64], label="RAM (GB)", value=8),
        gr.Number(label="Screen Size (inches)", value=15.6),
        gr.Number(label="Weight (kg)", value=2.0),
        gr.Dropdown(["1366x768", "1600x900", "1920x1080", "2560x1440", "2560x1600", "2880x1800", "3200x1800", "3840x2160"],
                    label="Screen Resolution", value="1920x1080"),
        gr.Checkbox(label="Touchscreen", value=False),
        gr.Checkbox(label="IPS Panel", value=False),
        gr.Dropdown(choices["Cpu_brand"], label="CPU", value="Intel Core i5"),
        gr.Number(label="CPU Speed (GHz)", value=2.5),
        gr.Dropdown([0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], label="SSD (GB)", value=256),
        gr.Dropdown([0, 500, 1024, 2048], label="HDD (GB)", value=0),
        gr.Dropdown(choices["Gpu_brand"], label="GPU Brand", value="Intel"),
        gr.Dropdown(choices["OS"], label="Operating System", value="Windows"),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Laptop Price Prediction App",
    description="Enter the main laptop specs to estimate its price. Other details are set to a typical laptop.",
)

if __name__ == "__main__":
    demo.launch()
