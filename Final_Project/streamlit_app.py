"""Road Accident Severity - Streamlit app (Final Project).
Loads the saved pipeline (imputer + scaler + encoder + model) and predicts Serious/Fatal vs Slight."""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_saved():
    return joblib.load(os.path.join(HERE, "models", "best_model.joblib"))


saved = load_saved()
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
    return p, label



st.set_page_config(page_title="Road Accident Severity", page_icon="🚦")
st.title("Road Accident Severity Prediction (UK 2018)")
st.write("Enter the conditions of a road accident. The model predicts whether it is Slight or Serious/Fatal. "
         "Details that are not asked are filled with typical values from the training data. For learning only, not for real decisions.")

c1, c2 = st.columns(2)
with c1:
    speed_limit = st.selectbox("Speed limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
    road_type = st.selectbox("Road type", list(ROAD_TYPE))
    road_class = st.selectbox("Road class", list(ROAD_CLASS), index=2)
    junction = st.selectbox("Junction", list(JUNCTION))
    area = st.selectbox("Area", list(AREA))
    light = st.selectbox("Light conditions", list(LIGHT))
    weather = st.selectbox("Weather", list(WEATHER))
    surface = st.selectbox("Road surface", list(SURFACE))
    day = st.selectbox("Day of the week", list(DAYS), index=5)
    hour = st.slider("Hour of the day", 0, 23, 17)
    month = st.slider("Month", 1, 12, 6)
with c2:
    n_vehicles = st.slider("Number of vehicles", 1, 10, 2)
    motorcycle = st.checkbox("A motorcycle is involved")
    pedal_cycle = st.checkbox("A pedal cycle is involved")
    bus = st.checkbox("A bus / coach is involved")
    heavy = st.checkbox("A heavy goods vehicle is involved")
    min_age = st.slider("Youngest driver age", 16, 90, 30)
    max_age = st.slider("Oldest driver age", 16, 90, 45)
    vehicle_age = st.slider("Age of the oldest vehicle (years)", 0, 30, 6)

if st.button("Predict severity", type="primary"):
    p, label = predict(speed_limit, road_type, road_class, junction, area, light, weather, surface, day, hour, month,
                       n_vehicles, motorcycle, pedal_cycle, bus, heavy, min_age, max_age, vehicle_age)
    st.metric("Risk of a Serious / Fatal accident", f"{p:.1%}")
    (st.error if label != "Slight" else st.success)(f"Predicted severity: {label} (decision threshold = {threshold:.2f})")
    st.progress(min(max(p, 0.0), 1.0))
