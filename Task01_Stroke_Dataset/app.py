"""Gradio app for the Stroke Prediction model (Task 01).

Run:  python app.py
Needs model/stroke_model.joblib (created by the last cells of notebook.ipynb).
"""
from pathlib import Path

import joblib
import pandas as pd

bundle = joblib.load(Path(__file__).parent / "model" / "stroke_model.joblib")
pipeline = bundle["pipeline"]        # preprocessing + trained model
THRESHOLD = bundle["threshold"]      # decision threshold chosen in the notebook
FEATURES = bundle["features"]


def predict(gender, age, hypertension, heart_disease, ever_married, work_type,
            residence_type, avg_glucose_level, bmi, smoking_status):
    row = pd.DataFrame([{
        "gender": gender,
        "age": float(age),
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": float(bmi),
        "smoking_status": smoking_status,
    }])[FEATURES]
    proba = float(pipeline.predict_proba(row)[0, 1])
    verdict = "HIGH RISK - recommend a medical check-up" if proba >= THRESHOLD else "Lower risk"
    text = (f"### {verdict}\nEstimated stroke probability: **{proba:.1%}**  \n"
            f"Model: {bundle['model_name']} (threshold {THRESHOLD:.2f})\n\n"
            "_Educational demo, not a medical diagnosis._")
    return {"Stroke": proba, "No stroke": 1 - proba}, text


def build_ui():
    import gradio as gr
    inputs = [
        gr.Radio(["Female", "Male"], value="Female", label="Gender"),
        gr.Slider(1, 100, value=50, step=1, label="Age"),
        gr.Radio(["No", "Yes"], value="No", label="Hypertension"),
        gr.Radio(["No", "Yes"], value="No", label="Heart disease"),
        gr.Radio(["Yes", "No"], value="Yes", label="Ever married"),
        gr.Dropdown(["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                    value="Private", label="Work type"),
        gr.Radio(["Urban", "Rural"], value="Urban", label="Residence type"),
        gr.Number(value=100.0, label="Average glucose level (mg/dL)"),
        gr.Number(value=27.0, label="BMI"),
        gr.Dropdown(["never smoked", "formerly smoked", "smokes", "Unknown"],
                    value="never smoked", label="Smoking status"),
    ]
    return gr.Interface(fn=predict, inputs=inputs,
                        outputs=[gr.Label(label="Prediction"), gr.Markdown()],
                        title="Stroke Risk Prediction",
                        description="Enter patient information to estimate the stroke risk.",
                        flagging_mode="never")


if __name__ == "__main__":
    build_ui().launch()
