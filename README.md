# ✈️ Aircraft Engine RUL Predictor

A solo, end-to-end machine learning project that predicts the **Remaining Useful
Life (RUL)** of aircraft engines from sensor data, and serves those predictions
through a web API and an interactive dashboard.

Built with the **NASA C-MAPSS (FD001)** dataset as a learning project to
understand a complete ML pipeline — from raw data all the way to a usable app.

---

## What it does

Given an aircraft engine's sensor readings, the system predicts how many operating
cycles the engine has left before failure, and translates that into a simple
maintenance risk level:

| Predicted RUL | Risk level | Meaning |
|---------------|-----------|---------|
| `< 30` cycles | 🔴 High | Inspect before next flight |
| `30 – 74` cycles | 🟠 Medium | Schedule maintenance soon |
| `>= 75` cycles | 🟢 Low | Routine monitoring |

The goal: help a maintenance team see **which engines to service first**, instead
of guessing or replacing on a fixed schedule.

---

## The three pieces

```
1. Notebook   →  loads data, builds the RUL target, trains a model, saves model.pkl
        ↓
2. Backend    →  FastAPI service that loads the model and answers prediction requests
        ↓
3. Dashboard  →  Streamlit page that sends readings to the backend and shows the result
```

Data flows in one direction: the dashboard collects sensor readings → sends them to
the backend → the backend runs the model → returns a prediction → the dashboard
displays it with a colour-coded risk level.

---

## Project structure

```
aircraft-rul-predictor/
├── notebooks/        # the training notebook (data → model)
├── backend/
│   └── main.py       # FastAPI app: loads model.pkl, exposes /predict
├── dashboard/
│   └── app.py        # Streamlit dashboard (the user-facing screen)
├── models/
│   ├── model.pkl              # trained model (scaler + RandomForest)
│   └── model_metadata.json    # feature list, metrics, risk thresholds
├── data/             # NASA C-MAPSS FD001 data
├── requirements.txt  # Python dependencies
└── README.md
```

---

## The model

- **Type:** scikit-learn `Pipeline` — `MinMaxScaler` + `RandomForestRegressor`
- **Inputs:** 24 raw features (3 operational settings + 21 sensors), one cycle at a time
- **Target:** RUL, clipped at 125 cycles
- **Split:** engine-level (whole engines held out for testing) to avoid data leakage

The model is trained in the notebook and exported as `model.pkl`. The backend loads
this file — it never retrains.

---

## How to run

All commands are run from the **repository root**.

### 1. Set up the environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the backend (terminal 1)

```bash
uvicorn backend.main:app --reload
```

- Health check: <http://127.0.0.1:8000>
- Interactive API docs: <http://127.0.0.1:8000/docs>

### 3. Start the dashboard (terminal 2)

Open a **second terminal** (keep the backend running in the first):

```bash
streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501>. Enter sensor readings, click
**Predict RUL**, and see the predicted life and risk level.

---

## Tech used

- **Python** · **pandas** / **numpy** — data handling
- **scikit-learn** — model training (RandomForest)
- **FastAPI** + **uvicorn** — the prediction API
- **Streamlit** — the dashboard
- **joblib** — saving / loading the model

---

## Notes

This is a personal learning project built to understand how a real predictive
maintenance pipeline fits together end to end. It uses the simple single-cycle
feature approach for clarity. Possible next steps: rolling-window features for
higher accuracy, model explainability (SHAP), and batch/fleet-level predictions.
