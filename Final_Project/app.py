"""Road Accident Severity - Gradio app (Final Project).
Loads the saved pipeline (imputer + scaler + encoder + model) and predicts Serious/Fatal vs Slight."""
import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
saved = joblib.load(os.path.join(HERE, "models", "best_model.joblib"))
pipeline, threshold, columns = saved["pipeline"], saved["threshold"], saved["feature_columns"]

# friendly names for the codes of the UK road safety data
ROAD_TYPE = {"Single carriageway": 6, "Dual carriageway": 3, "Roundabout": 1, "One way street": 2, "Slip road": 7}
LIGHT = {"Daylight": 1, "Darkness - street lights lit": 4, "Darkness - street lights unlit": 5, "Darkness - no street lighting": 6}
WEATHER = {"Fine, no high winds": 1, "Raining, no high winds": 2, "Snowing, no high winds": 3, "Fine, high winds": 4,
           "Raining, high winds": 5, "Snowing, high winds": 6, "Fog or mist": 7}
SURFACE = {"Dry": 1, "Wet or damp": 2, "Snow": 3, "Frost or ice": 4, "Flood (over 3cm)": 5, "Oil or diesel": 6, "Mud": 7}
AREA = {"Urban": 1, "Rural": 2}
JUNCTION = {"Not at a junction": 0, "Roundabout": 1, "Mini-roundabout": 2, "T or staggered junction": 3,
            "Slip road": 5, "Crossroads": 6, "More than 4 arms": 7, "Private drive": 8, "Other junction": 9}
ROAD_CLASS = {"Motorway": 1, "A(M)": 2, "A road": 3, "B road": 4, "C road": 5, "Unclassified": 6}
DAYS = {"Sunday": 1, "Monday": 2, "Tuesday": 3, "Wednesday": 4, "Thursday": 5, "Friday": 6, "Saturday": 7}


def predict(speed_limit, road_type, road_class, junction, area, light, weather, surface, day, hour, month,
            n_vehicles, motorcycle, pedal_cycle, bus, heavy, min_age, max_age, vehicle_age):
    row = {c: np.nan for c in columns}          # everything not asked stays missing -> the pipeline fills it (train values)
    row.update({
        "Speed_limit": speed_limit, "Road_Type": ROAD_TYPE[road_type], "1st_Road_Class": ROAD_CLASS[road_class],
        "Junction_Detail": JUNCTION[junction], "Urban_or_Rural_Area": AREA[area], "Light_Conditions": LIGHT[light],
        "Weather_Conditions": WEATHER[weather], "Road_Surface_Conditions": SURFACE[surface],
        "Day_of_Week": DAYS[day], "Is_Weekend": int(DAYS[day] in (1, 7)), "Hour": hour, "Month": month,
        "Number_of_Vehicles": n_vehicles, "Has_Motorcycle": int(motorcycle), "Has_Pedal_Cycle": int(pedal_cycle),
        "Has_Bus": int(bus), "Has_Heavy_Vehicle": int(heavy),
        "Min_Driver_Age": min_age, "Max_Driver_Age": max(min_age, max_age), "Max_Vehicle_Age": vehicle_age,
    })
    X = pd.DataFrame([row])[columns]
    p = float(pipeline.predict_proba(X)[0, 1])
    label = "Serious / Fatal" if p >= threshold else "Slight"
    return {"Serious / Fatal": p, "Slight": 1 - p}, f"Predicted severity: {label}  (risk = {p:.1%}, decision threshold = {threshold:.2f})"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown([20, 30, 40, 50, 60, 70], value=30, label="Speed limit (mph)"),
        gr.Dropdown(list(ROAD_TYPE), value="Single carriageway", label="Road type"),
        gr.Dropdown(list(ROAD_CLASS), value="A road", label="Road class"),
        gr.Dropdown(list(JUNCTION), value="Not at a junction", label="Junction"),
        gr.Dropdown(list(AREA), value="Urban", label="Area"),
        gr.Dropdown(list(LIGHT), value="Daylight", label="Light conditions"),
        gr.Dropdown(list(WEATHER), value="Fine, no high winds", label="Weather"),
        gr.Dropdown(list(SURFACE), value="Dry", label="Road surface"),
        gr.Dropdown(list(DAYS), value="Friday", label="Day of the week"),
        gr.Slider(0, 23, value=17, step=1, label="Hour of the day"),
        gr.Slider(1, 12, value=6, step=1, label="Month"),
        gr.Slider(1, 10, value=2, step=1, label="Number of vehicles"),
        gr.Checkbox(label="A motorcycle is involved", value=False),
        gr.Checkbox(label="A pedal cycle is involved", value=False),
        gr.Checkbox(label="A bus / coach is involved", value=False),
        gr.Checkbox(label="A heavy goods vehicle is involved", value=False),
        gr.Slider(16, 90, value=30, step=1, label="Youngest driver age"),
        gr.Slider(16, 90, value=45, step=1, label="Oldest driver age"),
        gr.Slider(0, 30, value=6, step=1, label="Age of the oldest vehicle (years)"),
    ],
    outputs=[gr.Label(label="Probability"), gr.Textbox(label="Prediction")],
    title="Road Accident Severity Prediction (UK 2018)",
    description="Enter the conditions of a road accident. The model predicts whether it is Slight or Serious/Fatal. "
                "Details that are not asked are filled with typical values from the training data. For learning only, not for real decisions.",
)

if __name__ == "__main__":
    demo.launch()
