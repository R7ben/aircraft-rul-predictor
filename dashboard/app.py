# app.py — a friendly Streamlit dashboard that calls our RUL backend.

import streamlit as st       # builds the web page
import requests              # talks to our backend

# The address where our backend is running.
BACKEND_URL = "http://127.0.0.1:8000/predict"

# --- Page setup ---
st.set_page_config(page_title="Aircraft RUL Predictor", page_icon="✈️")
st.title("✈️ Aircraft Engine Health Predictor")
st.write("Enter an engine's sensor readings and get its predicted Remaining Useful Life (RUL).")

# --- A healthy engine's readings, used to pre-fill the form ---
default_reading = {
    "op_setting_1": -0.0007, "op_setting_2": -0.0004, "op_setting_3": 100.0,
    "sensor_01": 518.67, "sensor_02": 641.82, "sensor_03": 1589.70, "sensor_04": 1400.60,
    "sensor_05": 14.62, "sensor_06": 21.61, "sensor_07": 554.36, "sensor_08": 2388.06,
    "sensor_09": 9046.19, "sensor_10": 1.30, "sensor_11": 47.47, "sensor_12": 521.66,
    "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195, "sensor_16": 0.03,
    "sensor_17": 392.0, "sensor_18": 2388.0, "sensor_19": 100.0, "sensor_20": 39.06,
    "sensor_21": 23.419,
}

# --- Build an input box for each of the 24 features ---
st.subheader("Sensor inputs")
reading = {}
cols = st.columns(3)   # arrange inputs in 3 tidy columns
for i, (name, value) in enumerate(default_reading.items()):
    with cols[i % 3]:
        reading[name] = st.number_input(name, value=float(value), format="%.4f")

# --- The predict button ---
if st.button("Predict RUL", type="primary"):
    try:
        # Send the readings to our backend and get the prediction back
        response = requests.post(BACKEND_URL, json=reading)
        result = response.json()

        rul = result["predicted_rul"]
        risk = result["risk_level"]

        # Show the RUL number
        st.metric("Predicted Remaining Useful Life", f"{rul} cycles")

        # Show the risk level with a colour
        if risk == "High":
            st.error(f"🔴 Risk level: {risk} — inspect before next flight.")
        elif risk == "Medium":
            st.warning(f"🟠 Risk level: {risk} — schedule maintenance soon.")
        else:
            st.success(f"🟢 Risk level: {risk} — routine monitoring.")

    except Exception as e:
        st.error("Could not reach the backend. Is it running at 127.0.0.1:8000?")
        st.write(e)