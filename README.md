# ✈️ Aircraft Engine RUL Predictor

> End-to-end machine learning project that predicts the **Remaining Useful Life
> (RUL)** of aircraft engines from sensor data, served through a FastAPI backend
> and an interactive **cockpit-themed** Streamlit dashboard.

<p align="center">
  <img src="docs/screenshots/hero.png" alt="RUL Mission Control cockpit dashboard" width="900" />
</p>

Built with the **NASA C-MAPSS (FD001)** dataset to understand a complete ML
pipeline — from raw data, to trained model, to a usable maintenance tool.

---

## Highlights

- **Cockpit-style dashboard** — dark HUD theme, neon-cyan accents, animated
  gauges, live-updating trend chart, ambient aircraft animation (Lottie).
- **4 focused pages** — Single Engine · Fleet Batch · Compare Engines · Why?
- **Batch fleet prediction** — drop a CSV of engines, get a sorted risk
  table + distribution histogram back.
- **Explainability** — top-15 feature importance chart so a maintainer can
  see *why* the model flagged an engine.
- **Full FastAPI backend** with health check, single prediction, batch
  prediction, and feature-importance endpoints — auto-generated OpenAPI docs.

---

## Screenshots

| Single Engine | Fleet Batch |
|:---:|:---:|
| ![Single Engine](docs/screenshots/single-engine.png) | ![Fleet Batch](docs/screenshots/fleet-batch.png) |
| **Compare Engines** | **Why? (Explainability)** |
| ![Compare](docs/screenshots/compare.png) | ![Why](docs/screenshots/why.png) |

---

## What it does

Given an aircraft engine's sensor readings, the system predicts how many operating
cycles the engine has left before failure, and translates that into a simple
maintenance risk level:

| Predicted RUL | Risk | Meaning |
|---|---|---|
| `< 30` cycles | 🔴 High | Inspect before next flight |
| `30 – 74` cycles | 🟠 Medium | Schedule maintenance soon |
| `≥ 75` cycles | 🟢 Low | Routine monitoring |

The goal: help a maintenance team see **which engines to service first**,
instead of guessing or replacing on a fixed schedule.

---

## Architecture

```
┌──────────────┐   sensor    ┌──────────────┐   POST /predict   ┌──────────────┐
│  Notebook    │──── data ──▶│  RandomForest│◀──────────────────│  Streamlit   │
│  (training)  │             │  + Scaler    │──── prediction ──▶│  Dashboard   │
└──────────────┘             │  (model.pkl) │                   │  (cockpit UI)│
                             └──────┬───────┘                   └──────┬───────┘
                                    │                                  │
                                    │  loaded once by                  │  http://localhost:8501
                                    ▼                                  │
                             ┌──────────────┐                          │
                             │  FastAPI     │◀─────────────────────────┘
                             │  backend     │
                             │  :8000       │
                             └──────────────┘
```

The **notebook** trains and exports `model.pkl`. The **backend** loads that
file once at startup and exposes `/predict`, `/predict_batch`, and
`/feature_importance`. The **dashboard** never touches the model directly —
it only calls the backend.

---

## Project structure

```
aircraft-rul-predictor/
├── notebooks/           # training notebook (data → model.pkl)
├── backend/
│   └── main.py          # FastAPI: /predict, /predict_batch, /feature_importance
├── dashboard/
│   ├── app.py           # Streamlit cockpit UI (4 pages)
│   └── lottie/          # local Lottie animation JSONs
├── models/
│   ├── model.pkl        # trained pipeline (MinMaxScaler + RandomForest)
│   └── model_metadata.json
├── data/                # NASA C-MAPSS FD001
├── docs/screenshots/    # README images
├── .streamlit/
│   └── config.toml      # cockpit dark theme
├── requirements.txt
├── HOW_TO_RUN_UI.md     # quick-start guide for the dashboard
└── README.md
```

---

## The model

- **Type:** scikit-learn `Pipeline` — `MinMaxScaler` + `RandomForestRegressor`
- **Inputs:** 24 raw features (3 operational settings + 21 sensors), one cycle at a time
- **Target:** RUL, clipped at 125 cycles
- **Split:** engine-level (whole engines held out for testing) to avoid data leakage
- **Metrics:** MAE ≈ 14.5 cycles · RMSE ≈ 19.2 · R² ≈ 0.78

The model is trained in the notebook and exported as `model.pkl`. The backend
loads it once — it never retrains at runtime.

---

## Quick start

All commands from the **repository root**.

```bash
# 1. Environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 2. Backend (terminal 1)
uvicorn backend.main:app --reload --port 8000
# health: http://127.0.0.1:8000     docs: http://127.0.0.1:8000/docs

# 3. Dashboard (terminal 2)
streamlit run dashboard/app.py
# opens at http://localhost:8501
```

You'll see a green **BACKEND ONLINE** dot in the header when both are talking.
Detailed walkthrough in [`HOW_TO_RUN_UI.md`](HOW_TO_RUN_UI.md).

---

## Tech

`Python` · `pandas` · `numpy` · `scikit-learn` · `joblib` ·
`FastAPI` · `uvicorn` · `Streamlit` · `Plotly` ·
`streamlit-option-menu` · `streamlit-lottie`

---

## Design

Cockpit UI was mocked up in Figma before implementation to lock in the visual
language first. Deep-blue-black background, neon-cyan accents, monospace HUD
type, gauge zones matching the risk thresholds. See
[the Figma file](https://www.figma.com/design/i0e8MmVlU8x714mvcv3Huk) for
mockups of all 4 pages.

---

## What's next

- Rolling-window features for higher accuracy
- Real-time streaming mode (auto-updating gauges)
- SHAP local explanations per prediction
- Docker Compose one-command deploy

---

## License

Personal learning project. Data © NASA C-MAPSS.
