# main.py — a tiny FastAPI backend that serves RUL predictions.

from fastapi import FastAPI          # the web framework (our "waiter")
from pydantic import BaseModel       # validates incoming data
import joblib                        # loads our saved model
import pandas as pd                  # arranges data for the model
from pathlib import Path

# --- 1. Create the app ---
app = FastAPI(title="Aircraft RUL Predictor")

# --- 2. Load the trained model once, when the server starts ---
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
model = joblib.load(MODEL_PATH)

# The 24 features the model expects, in the exact order it was trained on.
FEATURES = ['op_setting_1', 'op_setting_2', 'op_setting_3'] \
           + [f'sensor_{i:02d}' for i in range(1, 22)]

# --- 3. Define what an incoming request must look like ---
# The dashboard will send these 24 numbers; Pydantic checks they're all there.
class EngineReading(BaseModel):
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_01: float
    sensor_02: float
    sensor_03: float
    sensor_04: float
    sensor_05: float
    sensor_06: float
    sensor_07: float
    sensor_08: float
    sensor_09: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

# --- 4. A simple health check, so we can confirm the server is alive ---
@app.get("/")
def home():
    return {"status": "running", "message": "RUL backend is alive"}

# --- 5. The main prediction endpoint ---
@app.post("/predict")
def predict(reading: EngineReading):
    # Turn the incoming reading into a one-row table in the right column order
    row = pd.DataFrame([reading.dict()])[FEATURES]

    # Ask the model to predict the RUL
    predicted_rul = float(model.predict(row)[0])

    # Turn the number into a risk level (same thresholds as your team)
    if predicted_rul < 30:
        risk = "High"
    elif predicted_rul < 75:
        risk = "Medium"
    else:
        risk = "Low"

    # Send the answer back
    return {
        "predicted_rul": round(predicted_rul, 1),
        "risk_level": risk
    }