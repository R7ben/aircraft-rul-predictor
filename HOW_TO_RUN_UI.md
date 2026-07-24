# How to run the new cockpit UI

Follow these three steps from the repo root.

## 1. Install the new libraries

```bash
pip install -r requirements.txt
```

New additions vs. before: `plotly`, `streamlit-lottie`, `streamlit-option-menu`.

## 2. Start the backend (Terminal window #1)

```bash
uvicorn backend.main:app --reload --port 8000
```

Check it works by visiting <http://127.0.0.1:8000> — you should see
`{"status":"running","message":"RUL backend is alive"}`.

## 3. Start the dashboard (Terminal window #2)

```bash
streamlit run dashboard/app.py
```

Streamlit will open the dashboard automatically. In the top bar you'll see
a green blinking **BACKEND ONLINE** dot — that means the two apps found
each other.

---

## What each page does

| Page | What it shows |
|------|---------------|
| **Single Engine** | Enter 24 sensor values → animated cockpit gauge + risk badge + RUL trend line |
| **Fleet Batch** | Drag-drop a CSV of many engines → summary KPIs, histogram, sorted risk table, download results |
| **Compare Engines** | Two side-by-side engines with linked gauges + grouped bar comparison |
| **Why?** | Bar chart of the top-15 features the model relies on |

## What to tweak first (learning exercise)

- **Change the accent color** — open `.streamlit/config.toml`, change
  `primaryColor` (try `#FF6B6B` for a red-alert theme).
- **Add a new page** — copy an `elif page == "..."` block in `app.py`,
  add its name to the `option_menu(options=[...])` list.
- **Add a Lottie animation** — `pip install streamlit-lottie`, then use
  `streamlit_lottie.st_lottie(...)` inside a panel for a flying-plane loop.

## If something breaks

- **BACKEND OFFLINE red dot** → your uvicorn terminal isn't running or is
  on a different port. Fix by re-running step 2.
- **`ModuleNotFoundError`** → re-run `pip install -r requirements.txt`
  inside your activated venv.
- **CSV upload rejected** → download the template on the Fleet Batch
  page; your columns must match exactly.
