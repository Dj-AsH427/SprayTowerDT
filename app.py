HOW TO UPDATE YOUR GITHUB REPO & REDEPLOY ON STREAMLIT CLOUD
=============================================================

1. COPY THESE FILES TO YOUR LOCAL SprayTowerDT FOLDER:
   - app.py (updated version with month toggle)
   - sample_data_all.pkl (NEW - includes all 18k rows from Jan-May)
   - sample_data.pkl (still used, 20k rows)
   - model_artifact.pkl
   - requirements.txt
   - .streamlit/config.toml
   - README.md

2. RUN IN YOUR TERMINAL (inside SprayTowerDT folder):
   git add .
   git commit -m "Add month toggle to data subset selector, include all months in sample data"
   git push origin main

3. GO TO https://share.streamlit.io
   - Find your app "spraytowerdt-..."
   - Click "Rerun" (or it auto-redeploys in ~2 minutes)

4. TEST IN THE APP:
   Sidebar → "Data Subset for Dashboard"
   - Click "By month"
   - You should NOW see all 5 months: 2026-01, 2026-02, 2026-03, 2026-04, 2026-05
   - Select Jan → MAE = 0.809%
   - Select Feb → MAE = 0.602%
   - Select May → MAE = 0.044%

WHAT'S CHANGED:
===============
- sample_data_all.pkl now has ALL 18,542 steady-state rows (all months)
- App loads sample_data_all.pkl instead of just last 20k rows
- Month selector now shows all 5 months: Jan, Feb, Mar, Apr, May
- Metrics update dynamically when you toggle months
