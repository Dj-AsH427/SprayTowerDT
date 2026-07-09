# Grey-Box Spray Dryer Optimizer — v9

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `localhost:8501`. Light theme, three tabs, P50/P90 clearly labeled, color-coded spec alerts.

## Files
- `app.py` — the dashboard (Live Monitor / Optimizer & Recommendations / Model Performance)
- `model_artifact.pkl` — trained LightGBM moisture model, P10/P50/P90 quantile models, T_inlet safety model, feature list, training bounds (retrained fresh from your `cleaned_data.xlsx`, same pipeline as `GreyBox_v9_Final.ipynb`)
- `sample_data.pkl` — last 20,000 steady-state rows, used to populate the "pick a snapshot" selector and the performance charts
- `.streamlit/config.toml` — forces the light theme regardless of the viewer's system theme
- `requirements.txt`

## What's inside the app
- **Live Monitor** — pick any historical snapshot, see current setpoints, P10/P50/P90 moisture prediction with a color-coded in-spec/near-limit/out-of-spec badge, and a confidence-band chart around that point in time.
- **Optimizer & Recommendations** — sliders for CA fan, HP pump, and Grate speed. FD fan is *not* a slider — it's shown live, auto-computed as `FD_base × (CA / CA_base)` (proportional coupling to CA, per the v9 design). Live cost breakdown pie chart, a CA-speed sensitivity ("sweet spot") chart, a **"Find best recommendation"** button that runs the real 4-knob grid-search optimizer (T_inlet ≥ 300°C safety gate + moisture P90 ≤ 3.5% spec gate), and an **"Accept recommendation"** button that applies the result straight to the sliders.
- **Model Performance** — walk-forward held-out MAE by month, top-15 feature importance, predicted-vs-actual scatter, P10–P90 coverage check.

## Free hosting
Push these files to a GitHub repo, then connect it at [share.streamlit.io](https://share.streamlit.io) — no Node.js, no server setup.

## Note
The optimizer only returns a recommendation if a *safe* (T_inlet + moisture spec) setpoint exists within ±5 Hz (CA/HP) / ±3 Hz (Grate) of the current sliders. If your current point is already far outside the 3.0–3.5% spec, it's normal for no candidate to qualify — the app explains this in the UI when it happens.
