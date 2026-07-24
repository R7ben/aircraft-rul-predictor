# app.py — cockpit-style Streamlit dashboard for the Aircraft RUL Predictor.
#
# What each section does is called out with big comment banners so you can
# skim the file top-to-bottom and follow along.

import io
import json
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# ---- LOTTIE ANIMATIONS ---------------------------------------------------
# Prefers LOCAL files (offline-safe, faster) but falls back to a URL if given.
# To use: download a Lottie JSON from https://lottiefiles.com/ (Download →
# "Lottie JSON") and save it as dashboard/lottie/<name>.json — the filename
# must match the key below.
from pathlib import Path
LOTTIE_DIR = Path(__file__).parent / "lottie"

LOTTIE_SOURCES = {
    "aircraft": {"file": "aircraft.json", "url": None},
    "turbine":  {"file": "turbine.json",  "url": None},
}

@st.cache_data(ttl=3600)
def load_lottie(name: str) -> dict | None:
    """Load a Lottie animation by name. Returns None if it can't be found —
    the app still runs, it just falls back to a text placeholder."""
    src = LOTTIE_SOURCES.get(name, {})
    fp = LOTTIE_DIR / (src.get("file") or "")
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    url = src.get("url")
    if url:
        try:
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return None

# =============================================================================
# 1. PAGE SETUP + CONSTANTS
# =============================================================================
st.set_page_config(
    page_title="RUL Mission Control",
    page_icon="✈️",
    layout="wide",                       # use the full screen (cockpit-wide)
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"    # where FastAPI is running

FEATURES = ["op_setting_1", "op_setting_2", "op_setting_3"] + [
    f"sensor_{i:02d}" for i in range(1, 22)
]

DEFAULT_READING = {
    "op_setting_1": -0.0007, "op_setting_2": -0.0004, "op_setting_3": 100.0,
    "sensor_01": 518.67, "sensor_02": 641.82, "sensor_03": 1589.70, "sensor_04": 1400.60,
    "sensor_05": 14.62,  "sensor_06": 21.61,  "sensor_07": 554.36,  "sensor_08": 2388.06,
    "sensor_09": 9046.19,"sensor_10": 1.30,   "sensor_11": 47.47,   "sensor_12": 521.66,
    "sensor_13": 2388.02,"sensor_14": 8138.62,"sensor_15": 8.4195,  "sensor_16": 0.03,
    "sensor_17": 392.0,  "sensor_18": 2388.0, "sensor_19": 100.0,   "sensor_20": 39.06,
    "sensor_21": 23.419,
}

# Session state = a little memory that survives Streamlit's re-runs
if "history" not in st.session_state:
    st.session_state.history = []        # list of past predictions

# =============================================================================
# 2. CUSTOM CSS — turns Streamlit into a cockpit
# =============================================================================
st.markdown(
    """
    <style>
    /* ---- Global background: subtle radial glow like a HUD ---- */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0,229,255,0.08), transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(0,229,255,0.05), transparent 40%),
            #0A0F1C;
    }

    /* ---- The big title bar ---- */
    .cockpit-header {
        border: 1px solid rgba(0,229,255,0.35);
        border-radius: 12px;
        padding: 18px 22px;
        background: linear-gradient(90deg, rgba(0,229,255,0.08), rgba(0,229,255,0));
        box-shadow: 0 0 24px rgba(0,229,255,0.12) inset;
        margin-bottom: 18px;
    }
    .cockpit-header h1 {
        font-family: 'Courier New', monospace;
        color: #00E5FF;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 0 0 12px rgba(0,229,255,0.6);
    }
    .cockpit-header p {
        color: #9CA3AF; margin: 4px 0 0; letter-spacing: 1px;
    }

    /* ---- Reusable "panel" card, used everywhere ---- */
    .panel {
        border: 1px solid rgba(0,229,255,0.20);
        border-radius: 10px;
        padding: 14px 16px;
        background: rgba(17,24,39,0.6);
        box-shadow: 0 0 12px rgba(0,229,255,0.05) inset;
        margin-bottom: 14px;
    }
    .panel h3 { color: #00E5FF; margin-top: 0; letter-spacing: 1px; }

    /* ---- Little blinking "LIVE" dot ---- */
    .status-dot {
        display:inline-block; width:10px; height:10px; border-radius:50%;
        background:#22c55e; box-shadow:0 0 8px #22c55e;
        animation: pulse 1.4s infinite;
    }
    @keyframes pulse {
        0%   { opacity: 1; transform: scale(1);   }
        50%  { opacity: .4;transform: scale(1.4); }
        100% { opacity: 1; transform: scale(1);   }
    }

    /* ---- Primary button glow ---- */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg,#00E5FF,#0891B2);
        color:#001018; border:none; font-weight:700; letter-spacing:1px;
        box-shadow: 0 0 18px rgba(0,229,255,0.45);
        transition: transform .1s ease-in-out;
    }
    div.stButton > button[kind="primary"]:hover { transform: translateY(-1px); }

    /* ---- Sidebar polish ---- */
    section[data-testid="stSidebar"] {
        background: #060912;
        border-right: 1px solid rgba(0,229,255,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 3. BACKEND HELPERS  — small functions that call FastAPI
# =============================================================================
def backend_alive() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False

def predict_one(reading: dict) -> dict | None:
    try:
        r = requests.post(f"{BACKEND_URL}/predict", json=reading, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None

def predict_batch(rows: list[dict]) -> list[dict] | None:
    try:
        r = requests.post(f"{BACKEND_URL}/predict_batch",
                          json={"readings": rows}, timeout=30)
        r.raise_for_status()
        return r.json()["predictions"]
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None

@st.cache_data(ttl=300)
def get_feature_importance() -> pd.DataFrame:
    r = requests.get(f"{BACKEND_URL}/feature_importance", timeout=5)
    r.raise_for_status()
    return pd.DataFrame(r.json()["features"])

# =============================================================================
# 4. PLOTLY CHARTS — the "instruments"
# =============================================================================
def rul_gauge(rul: float) -> go.Figure:
    """Big cockpit-style gauge that shows predicted RUL."""
    if   rul < 30: bar_color = "#EF4444"   # red
    elif rul < 75: bar_color = "#F59E0B"   # amber
    else:          bar_color = "#22C55E"   # green

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rul,
        number={"suffix": " cyc", "font": {"size": 44, "color": "#E5E7EB"}},
        delta={"reference": 125, "decreasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 150], "tickcolor": "#00E5FF"},
            "bar": {"color": bar_color, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "rgba(0,229,255,0.4)",
            "steps": [
                {"range": [0, 30],   "color": "rgba(239,68,68,0.20)"},
                {"range": [30, 75],  "color": "rgba(245,158,11,0.20)"},
                {"range": [75, 150], "color": "rgba(34,197,94,0.20)"},
            ],
            "threshold": {
                "line": {"color": "#00E5FF", "width": 3},
                "thickness": 0.85,
                "value": rul,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E5E7EB", "family": "monospace"},
        height=320, margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig

def sensor_bar(reading: dict) -> go.Figure:
    """Horizontal bar chart of the 21 sensor values (quick eyeball view)."""
    sensor_keys = [k for k in reading if k.startswith("sensor_")]
    values = [reading[k] for k in sensor_keys]
    fig = go.Figure(go.Bar(
        x=values, y=sensor_keys, orientation="h",
        marker=dict(color=values, colorscale="Cividis",
                    line=dict(color="#00E5FF", width=0.5)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="monospace"),
        height=520, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor="rgba(0,229,255,0.10)"),
        yaxis=dict(autorange="reversed"),
    )
    return fig

def history_line(history: list[dict]) -> go.Figure:
    """Line chart of RUL across recent predictions (feels 'live')."""
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["predicted_rul"],
        mode="lines+markers",
        line=dict(color="#00E5FF", width=3),
        marker=dict(size=9, color=df["predicted_rul"],
                    colorscale="Turbo", showscale=False),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.08)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="monospace"),
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        title="RUL trend — recent predictions",
        xaxis=dict(title="Prediction #", gridcolor="rgba(0,229,255,0.10)"),
        yaxis=dict(title="RUL (cycles)", gridcolor="rgba(0,229,255,0.10)"),
    )
    return fig

# =============================================================================
# 5. HEADER + SIDEBAR NAV
# =============================================================================
alive = backend_alive()
status_html = (
    '<span class="status-dot"></span> <b style="color:#22C55E">BACKEND ONLINE</b>'
    if alive else
    '<span style="color:#EF4444">● BACKEND OFFLINE</span>'
)

head_left, head_right = st.columns([5, 1])
with head_left:
    st.markdown(
        f"""
        <div class="cockpit-header">
          <h1>✈ RUL MISSION CONTROL</h1>
          <p>NASA C-MAPSS FD001 &nbsp;·&nbsp; RandomForest engine &nbsp;·&nbsp; {status_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_right:
    # Small looping aircraft/turbine — ambient cockpit vibe.
    _hdr_anim = load_lottie("aircraft")
    if _hdr_anim:
        st_lottie(_hdr_anim, height=90, key="header_lottie",
                  speed=1, loop=True, quality="high")

with st.sidebar:
    st.markdown("### FLIGHT DECK")
    page = option_menu(
        menu_title=None,
        options=["Single Engine", "Fleet Batch", "Compare Engines", "Why?"],
        icons=["speedometer2", "cloud-upload", "columns-gap", "lightbulb"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background": "transparent"},
            "icon":      {"color": "#00E5FF", "font-size": "16px"},
            "nav-link":  {"color": "#9CA3AF", "font-family": "monospace",
                          "font-size": "14px", "margin": "4px 0",
                          "border-radius": "8px"},
            "nav-link-selected": {
                "background": "rgba(0,229,255,0.12)",
                "color": "#00E5FF",
                "border": "1px solid rgba(0,229,255,0.35)",
                "box-shadow": "0 0 12px rgba(0,229,255,0.25) inset",
            },
        },
    )
    st.markdown("---")
    st.caption("Backend: `127.0.0.1:8000`")
    if st.button("↺ Clear history"):
        st.session_state.history = []
        st.rerun()

# =============================================================================
# 6. PAGE: SINGLE ENGINE
# =============================================================================
if page == "Single Engine":
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="panel"><h3>◆ SENSOR INPUTS</h3>',
                    unsafe_allow_html=True)
        reading = {}
        cols = st.columns(3)
        for i, (name, val) in enumerate(DEFAULT_READING.items()):
            with cols[i % 3]:
                reading[name] = st.number_input(
                    name, value=float(val), format="%.4f", key=f"single_{name}"
                )
        go_btn = st.button("▶ RUN PREDICTION", type="primary",
                           use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><h3>◆ ENGINE STATUS</h3>',
                    unsafe_allow_html=True)
        if go_btn:
            # Big spinning turbine while we wait for the backend
            spin_slot = st.empty()
            _spin_anim = load_lottie("turbine")
            with spin_slot.container():
                if _spin_anim:
                    st_lottie(_spin_anim, height=220, key="predict_lottie",
                              speed=1.5, loop=True, quality="high")
                else:
                    st.markdown(
                        "<div style='text-align:center;color:#00E5FF;"
                        "font-family:monospace;font-size:20px;letter-spacing:2px;"
                        "padding:60px 0'>◌ ANALYZING TELEMETRY ◌</div>",
                        unsafe_allow_html=True,
                    )
                time.sleep(0.4)                # let the animation breathe
                result = predict_one(reading)
            spin_slot.empty()                  # remove animation once done
            if result:
                rul, risk = result["predicted_rul"], result["risk_level"]
                st.plotly_chart(rul_gauge(rul), use_container_width=True)
                colors = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}
                st.markdown(
                    f"<h2 style='text-align:center;color:{colors[risk]};"
                    f"text-shadow:0 0 12px {colors[risk]};letter-spacing:3px'>"
                    f"RISK: {risk.upper()}</h2>",
                    unsafe_allow_html=True,
                )
                # Remember this prediction for the history chart
                st.session_state.history.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "predicted_rul": rul,
                    "risk_level": risk,
                })
        else:
            st.info("Press **RUN PREDICTION** to see the gauge light up.")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.history:
            st.plotly_chart(history_line(st.session_state.history),
                            use_container_width=True)

    st.markdown('<div class="panel"><h3>◆ CURRENT SENSOR SNAPSHOT</h3>',
                unsafe_allow_html=True)
    st.plotly_chart(sensor_bar(reading), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 7. PAGE: FLEET BATCH (upload a CSV)
# =============================================================================
elif page == "Fleet Batch":
    st.markdown('<div class="panel"><h3>◆ FLEET BATCH PREDICT</h3>'
                '<p style="color:#9CA3AF">Upload a CSV with the 24 feature '
                'columns. Each row = one engine reading.</p>',
                unsafe_allow_html=True)

    up = st.file_uploader("Drop CSV here", type=["csv"])
    if up is None:
        # Offer a downloadable template so beginners aren't stuck
        template = pd.DataFrame([DEFAULT_READING])
        st.download_button(
            "⬇ Download CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="rul_template.csv",
            mime="text/csv",
        )
    else:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.dataframe(df.head(), use_container_width=True)
            if st.button("▶ PREDICT ALL", type="primary"):
                with st.spinner(f"Predicting {len(df)} engines..."):
                    rows = df[FEATURES].to_dict(orient="records")
                    preds = predict_batch(rows)
                if preds:
                    out = df.copy()
                    out["predicted_rul"] = [p["predicted_rul"] for p in preds]
                    out["risk_level"]    = [p["risk_level"]    for p in preds]

                    # Summary KPIs
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Engines",     len(out))
                    c2.metric("High risk",   int((out.risk_level == "High").sum()))
                    c3.metric("Medium risk", int((out.risk_level == "Medium").sum()))
                    c4.metric("Low risk",    int((out.risk_level == "Low").sum()))

                    # Distribution histogram
                    hist = go.Figure(go.Histogram(
                        x=out["predicted_rul"], nbinsx=20,
                        marker=dict(color="#00E5FF",
                                    line=dict(color="#0A0F1C", width=1)),
                    ))
                    hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E5E7EB", family="monospace"),
                        title="Fleet RUL distribution",
                        xaxis=dict(gridcolor="rgba(0,229,255,0.10)"),
                        yaxis=dict(gridcolor="rgba(0,229,255,0.10)"),
                        height=320, margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(hist, use_container_width=True)

                    # Result table, sorted highest-risk first
                    st.dataframe(
                        out.sort_values("predicted_rul").reset_index(drop=True),
                        use_container_width=True, height=380,
                    )
                    st.download_button(
                        "⬇ Download results",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="rul_predictions.csv",
                        mime="text/csv",
                    )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 8. PAGE: COMPARE ENGINES (side-by-side)
# =============================================================================
elif page == "Compare Engines":
    st.markdown('<div class="panel"><h3>◆ COMPARE ENGINES</h3>'
                '<p style="color:#9CA3AF">Tune two engines side-by-side and '
                'compare their predicted RUL live.</p></div>',
                unsafe_allow_html=True)

    colA, colB = st.columns(2)
    engines = {}
    for label, col in [("Engine A", colA), ("Engine B", colB)]:
        with col:
            st.markdown(f'<div class="panel"><h3>{label}</h3>',
                        unsafe_allow_html=True)
            r = {}
            # Give the user a lighter set of dials (op settings + 6 key sensors)
            key_features = ["op_setting_1", "op_setting_2", "op_setting_3",
                            "sensor_02", "sensor_03", "sensor_04",
                            "sensor_07", "sensor_11", "sensor_15"]
            for k in FEATURES:
                if k in key_features:
                    r[k] = st.number_input(
                        k, value=float(DEFAULT_READING[k]),
                        format="%.4f", key=f"{label}_{k}",
                    )
                else:
                    r[k] = DEFAULT_READING[k]   # keep the rest at healthy default
            engines[label] = r
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("▶ COMPARE", type="primary", use_container_width=True):
        results = {name: predict_one(r) for name, r in engines.items()}
        c1, c2 = st.columns(2)
        for (name, res), col in zip(results.items(), [c1, c2]):
            with col:
                if res:
                    st.plotly_chart(rul_gauge(res["predicted_rul"]),
                                    use_container_width=True)
                    st.markdown(
                        f"<h3 style='text-align:center;color:#00E5FF'>"
                        f"{name} — {res['risk_level'].upper()}</h3>",
                        unsafe_allow_html=True,
                    )

        # Overlay bar comparison of sensor readings
        rows = []
        for name, r in engines.items():
            for k, v in r.items():
                rows.append({"engine": name, "feature": k, "value": v})
        cmp_df = pd.DataFrame(rows)
        fig = go.Figure()
        for name in engines:
            sub = cmp_df[cmp_df.engine == name]
            fig.add_trace(go.Bar(
                x=sub.feature, y=sub.value, name=name,
                marker=dict(line=dict(color="#00E5FF", width=0.5)),
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB", family="monospace"),
            height=380, margin=dict(l=10, r=10, t=40, b=80),
            title="Feature-by-feature comparison",
            xaxis=dict(tickangle=-45, gridcolor="rgba(0,229,255,0.10)"),
            yaxis=dict(gridcolor="rgba(0,229,255,0.10)"),
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 9. PAGE: WHY?  (explainability — global feature importance)
# =============================================================================
elif page == "Why?":
    st.markdown('<div class="panel"><h3>◆ WHY DOES THE MODEL SAY THAT?</h3>'
                '<p style="color:#9CA3AF">These are the features the model '
                'leans on most when predicting RUL. Bigger bar = bigger '
                'influence.</p></div>',
                unsafe_allow_html=True)
    try:
        fi = get_feature_importance().head(15)
        fig = go.Figure(go.Bar(
            x=fi["importance"], y=fi["name"], orientation="h",
            marker=dict(color=fi["importance"], colorscale="Turbo",
                        line=dict(color="#00E5FF", width=0.5)),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB", family="monospace"),
            height=520, margin=dict(l=10, r=10, t=40, b=10),
            title="Top 15 most influential features",
            xaxis=dict(gridcolor="rgba(0,229,255,0.10)"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Full table"):
            st.dataframe(get_feature_importance(), use_container_width=True)
    except Exception as e:
        st.error(f"Couldn't load feature importance: {e}")
