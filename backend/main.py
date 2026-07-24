# main.py — FastAPI backend that serves RUL predictions.
# Extended: single predict, batch predict (from CSV rows), and feature importance.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # lets the dashboard talk to us
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. Create the app ---
app = FastAPI(title="Aircraft RUL Predictor")

# Allow the Streamlit dashboard (and any local tool) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Load the trained model once, when the server starts ---
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
model = joblib.load(MODEL_PATH)

# The 24 features the model expects, in the exact order it was trained on.
FEATURES = ['op_setting_1', 'op_setting_2', 'op_setting_3'] \
           + [f'sensor_{i:02d}' for i in range(1, 22)]

# --- 3. Request shape for a single reading ---
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

class BatchRequest(BaseModel):
    readings: List[EngineReading]

# --- Shared helper: turn a RUL number into a risk label ---
def risk_of(rul: float) -> str:
    if rul < 30:
        return "High"
    if rul < 75:
        return "Medium"
    return "Low"

# --- 4. Health check ---
@app.get("/")
def home():
    return {"status": "running", "message": "RUL backend is alive"}

# --- 5. Single prediction ---
@app.post("/predict")
def predict(reading: EngineReading):
    row = pd.DataFrame([reading.dict()])[FEATURES]
    predicted_rul = float(model.predict(row)[0])
    return {
        "predicted_rul": round(predicted_rul, 1),
        "risk_level": risk_of(predicted_rul),
    }

# --- 6. Batch prediction (for the CSV upload feature) ---
@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if not req.readings:
        raise HTTPException(status_code=400, detail="No readings provided")
    df = pd.DataFrame([r.dict() for r in req.readings])[FEATURES]
    preds = model.predict(df).astype(float)
    return {
        "predictions": [
            {"predicted_rul": round(float(p), 1), "risk_level": risk_of(float(p))}
            for p in preds
        ]
    }

# --- 7. Feature importance (for the "why did the model predict this?" panel) ---
@app.get("/feature_importance")
def feature_importance():
    # RandomForest lives inside a scikit-learn Pipeline; grab the last step
    try:
        rf = model.steps[-1][1] if hasattr(model, "steps") else model
        importances = rf.feature_importances_
    except AttributeError:
        raise HTTPException(status_code=500, detail="Model has no feature_importances_")
    ranked = sorted(
        zip(FEATURES, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {"features": [{"name": n, "importance": float(v)} for n, v in ranked]}
