"""
Grey-Box Spray Dryer Optimizer — v9
Interactive Streamlit dashboard wrapping the trained LightGBM moisture,
quantile (P10/P50/P90) and T_inlet safety models with the 4-knob cost
optimizer (CA fan, HP pump, Grate speed, FD fan coupled to CA).
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────
# CONFIG / CONSTANTS  (mirrors GreyBox_v9_Final.ipynb, cell 16)
# ─────────────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

MOISTURE_SPEC_HI = 3.5   # P90 must stay <= this
MOISTURE_SPEC_LO = 3.0   # P50 must stay >= this (avoid over-drying)
T_INLET_MIN = 300.0      # deg C safety floor

briq_cost_per_ton = 9650
elec_cost = 8.0
P_CA, CA_ref = 75.0, 30.0
P_HP, HP_ref = 55.0, 30.0
P_GR, GR_ref = 15.0, 14.0
P_FD, FD_ref = 37.0, 30.0

MAX_STEP_CA = 5
MAX_STEP_HP = 5
MAX_STEP_GR = 3

st.set_page_config(page_title="Grey-Box Spray Dryer Optimizer", page_icon="🌡️",
                    layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background-color: #FFFFFF; }
.badge {
    display:inline-block; padding: 6px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.95rem; color: white;
}
.badge-green  { background-color: #2E8B57; }
.badge-amber  { background-color: #E0912E; }
.badge-red    { background-color: #C0392B; }
.metric-card {
    background: #F4F6F8; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #E3E7EB;
}
.small-note { color:#5A6472; font-size:0.85rem; }
hr { margin: 0.6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_artifacts():
    artifact = joblib.load(os.path.join(APP_DIR, "model_artifact.pkl"))
    sample = pd.read_pickle(os.path.join(APP_DIR, "sample_data.pkl"))
    sample = sample.sort_values("Time and date").reset_index(drop=True)
    sample['month'] = sample['Time and date'].dt.to_period('M')
    return artifact, sample


artifact, sample = load_artifacts()


@st.cache_data
def find_default_snapshot(_features, _qmodels):
    """Pick a well-behaved starting snapshot where predicted moisture is
    already inside the 3.0-3.5% spec, so the optimizer demo has room to work with."""
    X = sample[_features].values
    p50 = _qmodels[0.50].predict(X)
    p90 = _qmodels[0.90].predict(X)
    in_spec = (p50 >= 3.0) & (p50 <= 3.5) & (p90 <= 3.6)
    idxs = np.where(in_spec)[0]
    if len(idxs) == 0:
        return len(sample) - 1
    return int(idxs[len(idxs) // 2])

final_model   = artifact["final_model"]
qmodels       = artifact["qmodels"]          # {0.10:, 0.50:, 0.90:}
tinlet_model  = artifact["tinlet_model"]
features      = artifact["features"]
tinlet_features = artifact["tinlet_features"]
controllable  = artifact["controllable"]
train_bounds  = artifact["train_bounds"]
feat_imp      = artifact["feature_importance"]
wf_df         = artifact["wf_results"]
grate_present = artifact["grate_present"]

ca_idx  = features.index("CA_Fan_Speed")
hp_idx  = features.index("Combind_HP_Pump_Speed")
fd_idx  = features.index("FD_Fan_speed")
ti_idx  = features.index("Tower_Inlet_Temp")
gr_idx  = features.index("Tower_Grade_Speed_Hz") if grate_present else None

ti_ca_idx = tinlet_features.index("CA_Fan_Speed") if "CA_Fan_Speed" in tinlet_features else None
ti_hp_idx = tinlet_features.index("Combind_HP_Pump_Speed") if "Combind_HP_Pump_Speed" in tinlet_features else None
ti_fd_idx = tinlet_features.index("FD_Fan_speed") if "FD_Fan_speed" in tinlet_features else None
ti_gr_idx = tinlet_features.index("Tower_Grade_Speed_Hz") if grate_present and "Tower_Grade_Speed_Hz" in tinlet_features else None


# ─────────────────────────────────────────────────────────────────────────
# CORE MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────
def cost_breakdown(ca, hp, gr, fd, briq_rate_kg_s):
    c_briq = briq_rate_kg_s * 3600 * briq_cost_per_ton / 1000
    c_ca = P_CA * (ca / CA_ref) ** 3 * elec_cost
    c_hp = P_HP * (hp / HP_ref) ** 3 * elec_cost
    c_gr = P_GR * (gr / GR_ref) ** 3 * elec_cost if grate_present else 0.0
    c_fd = P_FD * (fd / FD_ref) ** 3 * elec_cost
    total = c_briq + c_ca + c_hp + c_gr + c_fd
    return total, {"Briquette fuel": c_briq, "CA fan": c_ca, "HP pump": c_hp,
                    "Grate motor": c_gr, "FD fan": c_fd}


def predict_tinlet_single(base_row, ca, hp, gr, fd):
    vec = base_row[tinlet_features].values.reshape(1, -1).astype(float)
    if ti_ca_idx is not None: vec[0, ti_ca_idx] = ca
    if ti_hp_idx is not None: vec[0, ti_hp_idx] = hp
    if ti_fd_idx is not None: vec[0, ti_fd_idx] = fd
    if ti_gr_idx is not None: vec[0, ti_gr_idx] = gr
    return float(tinlet_model.predict(vec)[0])


def predict_moisture_single(base_row, ca, hp, gr, fd, ti_pred):
    vec = base_row[features].values.reshape(1, -1).astype(float)
    vec[0, ca_idx] = ca
    vec[0, hp_idx] = hp
    vec[0, fd_idx] = fd
    if gr_idx is not None: vec[0, gr_idx] = gr
    vec[0, ti_idx] = ti_pred
    p10 = float(qmodels[0.10].predict(vec)[0])
    p50 = float(qmodels[0.50].predict(vec)[0])
    p90 = float(qmodels[0.90].predict(vec)[0])
    return p10, p50, p90


def fd_from_ca(ca, cur_ca, cur_fd):
    """FD fan is coupled proportionally to the CA fan (per the v9 design)."""
    if cur_ca <= 0:
        return cur_fd
    return cur_fd * (ca / cur_ca)


def run_optimizer(base_row, cur_ca, cur_hp, cur_gr, cur_fd, briq_rate):
    """Vectorised replica of the notebook's 4-knob grid search (cell 16)."""
    ca_lo = max(15, cur_ca - MAX_STEP_CA); ca_hi = min(35, cur_ca + MAX_STEP_CA)
    hp_lo = max(5, cur_hp - MAX_STEP_HP);  hp_hi = min(45, cur_hp + MAX_STEP_HP)
    if grate_present:
        gr_lo = max(5, cur_gr - MAX_STEP_GR); gr_hi = min(22, cur_gr + MAX_STEP_GR)
        gr_grid = np.arange(gr_lo, gr_hi + 1)
    else:
        gr_grid = np.array([cur_gr])

    ca_grid = np.arange(ca_lo, ca_hi + 1)
    hp_grid = np.arange(hp_lo, hp_hi + 1)

    CA, HP, GR = np.meshgrid(ca_grid, hp_grid, gr_grid, indexing="ij")
    CA, HP, GR = CA.ravel(), HP.ravel(), GR.ravel()
    FD = cur_fd * (CA / cur_ca) if cur_ca > 0 else np.full_like(CA, cur_fd)
    n = len(CA)

    # bounds guard (extrapolation) using train P5-P95
    def within(colname, vals):
        lo, hi = train_bounds[colname]
        return (vals >= lo) & (vals <= hi)

    ok = within("CA_Fan_Speed", CA) & within("Combind_HP_Pump_Speed", HP) & within("FD_Fan_speed", FD)
    if grate_present:
        ok &= within("Tower_Grade_Speed_Hz", GR)
    if not ok.any():
        return None

    CA, HP, GR, FD = CA[ok], HP[ok], GR[ok], FD[ok]
    n = len(CA)

    # T_inlet prediction (batch)
    ti_base = np.tile(base_row[tinlet_features].values.astype(float), (n, 1))
    if ti_ca_idx is not None: ti_base[:, ti_ca_idx] = CA
    if ti_hp_idx is not None: ti_base[:, ti_hp_idx] = HP
    if ti_fd_idx is not None: ti_base[:, ti_fd_idx] = FD
    if ti_gr_idx is not None: ti_base[:, ti_gr_idx] = GR
    ti_pred = tinlet_model.predict(ti_base)

    safe = ti_pred >= T_INLET_MIN
    if not safe.any():
        return None
    CA, HP, GR, FD, ti_pred = CA[safe], HP[safe], GR[safe], FD[safe], ti_pred[safe]
    n = len(CA)

    # Moisture prediction (batch)
    mfeat = np.tile(base_row[features].values.astype(float), (n, 1))
    mfeat[:, ca_idx] = CA
    mfeat[:, hp_idx] = HP
    mfeat[:, fd_idx] = FD
    if gr_idx is not None: mfeat[:, gr_idx] = GR
    mfeat[:, ti_idx] = ti_pred

    p90 = qmodels[0.90].predict(mfeat)
    p50 = qmodels[0.50].predict(mfeat)

    spec_ok = (p90 <= MOISTURE_SPEC_HI) & (p50 >= MOISTURE_SPEC_LO)
    if not spec_ok.any():
        return None
    CA, HP, GR, FD, ti_pred, p50, p90 = (CA[spec_ok], HP[spec_ok], GR[spec_ok],
                                          FD[spec_ok], ti_pred[spec_ok], p50[spec_ok], p90[spec_ok])

    totals = np.array([cost_breakdown(ca, hp, gr, fd, briq_rate)[0]
                        for ca, hp, gr, fd in zip(CA, HP, GR, FD)])
    best_i = int(np.argmin(totals))
    return {
        "CA": float(CA[best_i]), "HP": float(HP[best_i]), "Grate": float(GR[best_i]),
        "FD": float(FD[best_i]), "T_inlet_pred": float(ti_pred[best_i]),
        "P50": float(p50[best_i]), "P90": float(p90[best_i]), "cost": float(totals[best_i]),
        "n_candidates": int(n),
    }


def spec_badge(p50, p90):
    if p90 <= MOISTURE_SPEC_HI and p50 >= MOISTURE_SPEC_LO:
        return '<span class="badge badge-green">✅ IN SPEC (3.0–3.5%)</span>'
    elif p90 <= MOISTURE_SPEC_HI + 0.3:
        return '<span class="badge badge-amber">⚠️ NEAR LIMIT</span>'
    else:
        return '<span class="badge badge-red">🚨 OUT OF SPEC</span>'


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────
if "base_idx" not in st.session_state:
    st.session_state.base_idx = find_default_snapshot(features, qmodels)
if "ca" not in st.session_state:
    row0 = sample.iloc[st.session_state.base_idx]
    st.session_state.ca = float(row0["CA_Fan_Speed"])
    st.session_state.hp = float(row0["Combind_HP_Pump_Speed"])
    st.session_state.gr = float(row0["Tower_Grade_Speed_Hz"]) if grate_present else 14.0
    st.session_state.fd_base = float(row0["FD_Fan_speed"])
    st.session_state.ca_base = float(row0["CA_Fan_Speed"])
if "accept_msg" not in st.session_state:
    st.session_state.accept_msg = None


def set_base(idx):
    st.session_state.base_idx = idx
    row = sample.iloc[idx]
    st.session_state.ca = float(row["CA_Fan_Speed"])
    st.session_state.hp = float(row["Combind_HP_Pump_Speed"])
    st.session_state.gr = float(row["Tower_Grade_Speed_Hz"]) if grate_present else 14.0
    st.session_state.fd_base = float(row["FD_Fan_speed"])
    st.session_state.ca_base = float(row["CA_Fan_Speed"])
    st.session_state.accept_msg = None


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌡️ Grey-Box Spray Dryer")
    st.caption("v9 — physics-informed moisture model + 4-knob cost optimizer")
    st.markdown("---")
    
    st.markdown("**Data Subset for Dashboard**")
    data_subset = st.radio(
        "View predictions from:",
        options=["All steady-state data", "Last 20,000 rows", "By month"],
        index=0,
        help="Switch between different data slices to see model performance vary"
    )
    
    if data_subset == "All steady-state data":
        display_sample = sample
        subset_desc = "All steady-state (18,542 rows)"
    elif data_subset == "Last 20,000 rows":
        display_sample = sample.tail(20000)
        subset_desc = "Recent data (last 20,000 rows)"
    else:  # By month
        month_sel = st.selectbox("Pick a month:", 
                                  sorted(sample['month'].unique(), reverse=True),
                                  format_func=lambda x: str(x))
        display_sample = sample[sample['month'] == month_sel]
        subset_desc = f"Month: {month_sel}"
    
    st.caption(f"**Current:** {subset_desc}")
    st.markdown("---")
    ts_options = display_sample["Time and date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    sel = st.selectbox("Pick a snapshot from current subset:", 
                        options=list(range(len(display_sample))),
                        index=min(st.session_state.base_idx, len(display_sample)-1),
                        format_func=lambda i: ts_options[i] if i < len(ts_options) else "N/A")
    if sel != st.session_state.base_idx:
        st.session_state.base_idx = sel
        st.session_state.ca = float(display_sample.iloc[sel]["CA_Fan_Speed"])
        st.session_state.hp = float(display_sample.iloc[sel]["Combind_HP_Pump_Speed"])
        st.session_state.gr = float(display_sample.iloc[sel]["Tower_Grade_Speed_Hz"]) if grate_present else 14.0
        st.session_state.fd_base = float(display_sample.iloc[sel]["FD_Fan_speed"])
        st.session_state.ca_base = float(display_sample.iloc[sel]["CA_Fan_Speed"])
        st.session_state.accept_msg = None
    if st.button("↻ Jump to latest in subset", use_container_width=True):
        st.session_state.base_idx = len(display_sample) - 1
        row_latest = display_sample.iloc[-1]
        st.session_state.ca = float(row_latest["CA_Fan_Speed"])
        st.session_state.hp = float(row_latest["Combind_HP_Pump_Speed"])
        st.session_state.gr = float(row_latest["Tower_Grade_Speed_Hz"]) if grate_present else 14.0
        st.session_state.fd_base = float(row_latest["FD_Fan_speed"])
        st.session_state.ca_base = float(row_latest["CA_Fan_Speed"])
        st.session_state.accept_msg = None
        st.rerun()
    st.markdown("---")
    st.caption("Model: LightGBM · Walk-forward validated")
    st.caption(f"Held-out MAE range: {wf_df['mae'].min():.2f}% – {wf_df['mae'].max():.2f}%")

base_row = display_sample.iloc[st.session_state.base_idx]

tab1, tab2, tab3 = st.tabs(["📊 Live Monitor", "🎛️ Optimizer & Recommendations", "📈 Model Performance"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Current operating snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CA Fan Speed (Hz)", f"{base_row['CA_Fan_Speed']:.1f}")
    c2.metric("HP Pump Speed (Hz)", f"{base_row['Combind_HP_Pump_Speed']:.1f}")
    c3.metric("Grate Speed (Hz)", f"{base_row['Tower_Grade_Speed_Hz']:.1f}" if grate_present else "n/a")
    c4.metric("FD Fan Speed (Hz)", f"{base_row['FD_Fan_speed']:.1f}")
    c5.metric("Tower Inlet Temp (°C)", f"{base_row['Tower_Inlet_Temp']:.0f}")

    ti_now = float(base_row["Tower_Inlet_Temp"])
    p10n, p50n, p90n = predict_moisture_single(base_row, base_row["CA_Fan_Speed"],
                                                base_row["Combind_HP_Pump_Speed"],
                                                base_row["Tower_Grade_Speed_Hz"] if grate_present else 14.0,
                                                base_row["FD_Fan_speed"], ti_now)
    actual_moist = float(base_row["Tower_Powder_Moisture"])

    st.markdown("---")
    mc1, mc2 = st.columns([1, 2])
    with mc1:
        st.markdown("#### Moisture prediction")
        st.markdown(spec_badge(p50n, p90n), unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
        <div class="metric-card">
        <b>P10 (optimistic):</b> {p10n:.2f}%<br>
        <b>P50 (median):</b> {p50n:.2f}%<br>
        <b>P90 (worst-case):</b> {p90n:.2f}%<br>
        <hr>
        <b>Actual (logged):</b> {actual_moist:.2f}%
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="small-note">Ambient {base_row["Ambient_temperature_of_Air"]:.1f}°C · '
                    f'RH {base_row["Relative_Humidity"]:.0f}% · '
                    f'Outlet {base_row["Tower_Outlet_Temp"]:.0f}°C</p>', unsafe_allow_html=True)

    with mc2:
        window = 300
        i0 = max(0, st.session_state.base_idx - window)
        i1 = min(len(sample), st.session_state.base_idx + window)
        win = sample.iloc[i0:i1]
        X_win = win[features].values
        p10w = qmodels[0.10].predict(X_win)
        p50w = qmodels[0.50].predict(X_win)
        p90w = qmodels[0.90].predict(X_win)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=win["Time and date"], y=p90w, line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=win["Time and date"], y=p10w, fill="tonexty",
                                  fillcolor="rgba(30,96,145,0.18)", line=dict(width=0),
                                  name="P10–P90 band"))
        fig.add_trace(go.Scatter(x=win["Time and date"], y=p50w, line=dict(color="#1E6091", width=2),
                                  name="Predicted (P50)"))
        fig.add_trace(go.Scatter(x=win["Time and date"], y=win["Tower_Powder_Moisture"],
                                  line=dict(color="#C0392B", width=1), opacity=0.7, name="Actual"))
        fig.add_hrect(y0=MOISTURE_SPEC_LO, y1=MOISTURE_SPEC_HI, fillcolor="#2E8B57",
                       opacity=0.08, line_width=0, annotation_text="spec band", annotation_position="top left")
        fig.add_vline(x=base_row["Time and date"], line_dash="dash", line_color="#888")
        fig.update_layout(title="Moisture — confidence band around selected snapshot",
                           template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10),
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — OPTIMIZER & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Adjust setpoints and see the predicted effect")
    st.caption("FD Fan is not directly adjustable — it is mechanically coupled to CA Fan "
               "(FD = FD_base × CA/CA_base), matching the plant's air-mover design.")

    ca_bounds = train_bounds["CA_Fan_Speed"]
    hp_bounds = train_bounds["Combind_HP_Pump_Speed"]
    gr_bounds = train_bounds.get("Tower_Grade_Speed_Hz", (6.0, 22.0))

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        ca_val = st.slider("CA Fan Speed (Hz)", float(max(10, ca_bounds[0]-3)), float(min(38, ca_bounds[1]+3)),
                            float(st.session_state.ca), 0.5, key="ca_slider")
    with sc2:
        hp_val = st.slider("HP Pump Speed (Hz)", float(max(0, hp_bounds[0]-3)), float(min(48, hp_bounds[1]+3)),
                            float(st.session_state.hp), 0.5, key="hp_slider")
    with sc3:
        if grate_present:
            gr_val = st.slider("Grate Speed (Hz)", float(max(3, gr_bounds[0]-2)), float(min(25, gr_bounds[1]+2)),
                                float(st.session_state.gr), 0.5, key="gr_slider")
        else:
            gr_val = 14.0
            st.info("Grate speed not present in this dataset.")

    fd_val = fd_from_ca(ca_val, st.session_state.ca_base, st.session_state.fd_base)
    st.markdown(f"**FD Fan Speed (auto, coupled to CA):** `{fd_val:.1f} Hz`  "
                f"<span class='small-note'>(base FD {st.session_state.fd_base:.1f} Hz × "
                f"{ca_val:.1f}/{st.session_state.ca_base:.1f})</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Live prediction for the slider values
    ti_live = predict_tinlet_single(base_row, ca_val, hp_val, gr_val, fd_val)
    p10l, p50l, p90l = predict_moisture_single(base_row, ca_val, hp_val, gr_val, fd_val, ti_live)
    briq_rate = float(base_row.get("briq_feed_kg_per_s", 1.5))
    W_evap = float(base_row.get("W_evap_kg_per_s", 2.5))
    live_total, live_parts = cost_breakdown(ca_val, hp_val, gr_val, fd_val, briq_rate)
    live_cpkg = live_total / (W_evap * 3600) if W_evap > 0 else float("nan")

    lc1, lc2, lc3 = st.columns([1, 1, 1.3])
    with lc1:
        st.markdown("#### Predicted outcome")
        st.markdown(spec_badge(p50l, p90l), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
        <b>P10:</b> {p10l:.2f}% &nbsp; <b>P50:</b> {p50l:.2f}% &nbsp; <b>P90:</b> {p90l:.2f}%<br>
        <b>Predicted T_inlet:</b> {ti_live:.0f} °C {"✅" if ti_live>=T_INLET_MIN else "🚨 below safety floor"}
        </div>
        """, unsafe_allow_html=True)
    with lc2:
        st.markdown("#### Cost")
        st.metric("Total energy + fuel cost", f"₹{live_total:,.0f}/hr")
        st.metric("₹ per kg water evaporated", f"₹{live_cpkg:.2f}/kg")
    with lc3:
        pie = go.Figure(data=[go.Pie(labels=list(live_parts.keys()), values=list(live_parts.values()),
                                      hole=0.45, marker=dict(colors=["#E07A1E", "#1E6091", "#2E8B6F", "#6B4C9A", "#C03030"]))])
        pie.update_layout(title="Cost breakdown", template="plotly_white", height=260,
                           margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
        st.plotly_chart(pie, use_container_width=True)

    st.markdown("---")
    oc1, oc2 = st.columns([1, 2])
    with oc1:
        st.markdown("#### Run the 4-knob optimizer")
        st.caption(f"Searches ±{MAX_STEP_CA} Hz (CA/HP), ±{MAX_STEP_GR} Hz (Grate) around the CURRENT sliders, "
                   "keeps T_inlet ≥ 300°C and moisture P90 ≤ 3.5%, picks the cheapest safe option.")
        run = st.button("🔍 Find best recommendation", use_container_width=True, type="primary")
        if run:
            with st.spinner("Searching setpoint space..."):
                rec = run_optimizer(base_row, ca_val, hp_val, gr_val, fd_val, briq_rate)
            st.session_state.last_rec = rec
            st.session_state.last_rec_cur_cost = live_total

        if st.session_state.get("last_rec") is not None:
            rec = st.session_state.last_rec
            saving = max(st.session_state.last_rec_cur_cost - rec["cost"], 0)
            st.success(f"Evaluated {rec['n_candidates']} safe candidates.")
            st.markdown(f"""
            <div class="metric-card">
            <b>Recommended CA:</b> {rec['CA']:.0f} Hz &nbsp;|&nbsp; <b>HP:</b> {rec['HP']:.0f} Hz
            &nbsp;|&nbsp; <b>Grate:</b> {rec['Grate']:.0f} Hz &nbsp;|&nbsp; <b>FD:</b> {rec['FD']:.1f} Hz<br>
            <b>Predicted P50/P90:</b> {rec['P50']:.2f}% / {rec['P90']:.2f}%<br>
            <b>Predicted T_inlet:</b> {rec['T_inlet_pred']:.0f} °C<br>
            <b>Cost:</b> ₹{rec['cost']:,.0f}/hr &nbsp; <b>Saving vs current:</b> ₹{saving:,.0f}/hr
            </div>
            """, unsafe_allow_html=True)

            if st.button("✅ Accept recommendation — apply to sliders", use_container_width=True):
                st.session_state.ca = rec["CA"]
                st.session_state.hp = rec["HP"]
                st.session_state.gr = rec["Grate"]
                st.session_state.ca_base = rec["CA"]
                st.session_state.fd_base = rec["FD"]
                st.session_state.accept_msg = (f"Recommendation accepted at "
                                                f"{pd.Timestamp.now().strftime('%H:%M:%S')} — "
                                                f"setpoints updated (CA {rec['CA']:.0f}, HP {rec['HP']:.0f}, "
                                                f"Grate {rec['Grate']:.0f} Hz).")
                st.session_state.last_rec = None
                st.rerun()
        elif run:
            st.warning("No safe candidate found within the search window: every nearby setpoint either "
                       "breaches the T_inlet ≥ 300°C safety floor, the moisture spec, or the training-data "
                       "extrapolation bounds. This usually means the current point is already far outside "
                       "spec — try a snapshot closer to the 3.0–3.5% band, or widen the sliders manually.")
        else:
            st.info("Click the button above to search for a cheaper, safe setpoint near your current sliders.")

    with oc2:
        # comparison bar chart: current vs recommended (if any)
        if st.session_state.get("accept_msg"):
            st.success(st.session_state.accept_msg)
        rec = st.session_state.get("last_rec")
        labels = ["CA", "HP", "Grate", "FD"]
        cur_vals = [ca_val, hp_val, gr_val, fd_val]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Current", x=labels, y=cur_vals, marker_color="#1E6091"))
        if rec is not None:
            rec_vals = [rec["CA"], rec["HP"], rec["Grate"], rec["FD"]]
            fig2.add_trace(go.Bar(name="Recommended", x=labels, y=rec_vals, marker_color="#2E8B57"))
        fig2.update_layout(barmode="group", template="plotly_white", height=300, title="Setpoints — current vs recommended",
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        # moisture band vs candidate CA sweep (visual "sweet spot")
        sweep_ca = np.linspace(max(15, ca_val-8), min(35, ca_val+8), 25)
        sweep_fd = fd_from_ca(sweep_ca, st.session_state.ca_base, st.session_state.fd_base)
        sweep_ti = np.array([predict_tinlet_single(base_row, c, hp_val, gr_val, f) for c, f in zip(sweep_ca, sweep_fd)])
        sweep_p10, sweep_p50, sweep_p90 = [], [], []
        for c, f, t in zip(sweep_ca, sweep_fd, sweep_ti):
            a, b, d = predict_moisture_single(base_row, c, hp_val, gr_val, f, t)
            sweep_p10.append(a); sweep_p50.append(b); sweep_p90.append(d)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sweep_ca, y=sweep_p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=sweep_ca, y=sweep_p10, fill="tonexty", fillcolor="rgba(30,96,145,0.18)",
                                   line=dict(width=0), name="P10–P90 band"))
        fig3.add_trace(go.Scatter(x=sweep_ca, y=sweep_p50, line=dict(color="#1E6091"), name="P50"))
        fig3.add_hrect(y0=MOISTURE_SPEC_LO, y1=MOISTURE_SPEC_HI, fillcolor="#2E8B57", opacity=0.08, line_width=0)
        fig3.add_vline(x=ca_val, line_dash="dash", line_color="#888", annotation_text="current CA")
        fig3.update_layout(title="Moisture sensitivity to CA fan speed (holding HP/Grate fixed)",
                            template="plotly_white", height=300, margin=dict(l=10, r=10, t=40, b=10),
                            xaxis_title="CA Fan Speed (Hz)", yaxis_title="Moisture (%)")
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Walk-forward validated performance")
    p1, p2 = st.columns(2)
    with p1:
        figwf = px.bar(wf_df, x="month", y="mae", text_auto=".3f",
                        title="Held-out MAE by month (walk-forward CV)", template="plotly_white",
                        color_discrete_sequence=["#1E6091"])
        figwf.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="MAE (%)")
        st.plotly_chart(figwf, use_container_width=True)
    with p2:
        top15 = feat_imp.head(15).sort_values()
        figimp = go.Figure(go.Bar(x=top15.values, y=top15.index, orientation="h",
                                   marker_color=["#2E8B57" if f in controllable else "#1E6091" for f in top15.index]))
        figimp.update_layout(title="Top 15 feature importance (gain) — green = controllable",
                              template="plotly_white", height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figimp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Predicted (P50) vs actual — current data subset")
    X_subset = display_sample[features].values
    p50_subset = qmodels[0.50].predict(X_subset)
    p90_subset = qmodels[0.90].predict(X_subset)
    in_spec = (p90_subset <= MOISTURE_SPEC_HI) & (p50_subset >= MOISTURE_SPEC_LO)
    scatter_df = pd.DataFrame({
        "Actual": display_sample["Tower_Powder_Moisture"].values,
        "Predicted (P50)": p50_subset,
        "In spec": np.where(in_spec, "In spec", "Out of spec"),
    })
    figsc = px.scatter(scatter_df, x="Actual", y="Predicted (P50)", color="In spec",
                        opacity=0.35, template="plotly_white",
                        color_discrete_map={"In spec": "#2E8B57", "Out of spec": "#C0392B"})
    lims = [scatter_df[["Actual", "Predicted (P50)"]].min().min(), scatter_df[["Actual", "Predicted (P50)"]].max().max()]
    figsc.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="#888"), name="perfect fit"))
    figsc.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figsc, use_container_width=True)

    mae_subset = float(np.mean(np.abs(scatter_df["Actual"] - scatter_df["Predicted (P50)"])))
    p10_subset = qmodels[0.10].predict(X_subset)
    coverage = float(((display_sample["Tower_Powder_Moisture"].values >= p10_subset) &
                       (display_sample["Tower_Powder_Moisture"].values <= p90_subset)).mean())
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (this subset)", f"{mae_subset:.3f}%")
    m2.metric("P10–P90 coverage", f"{100*coverage:.1f}%", help="Target ≈ 80% for a calibrated band")
    m3.metric("Rows in subset", f"{len(display_sample):,}")

    st.markdown("---")
    st.markdown("""
    <div class="small-note">
    <b>Model card</b> — LightGBM gradient-boosted trees, walk-forward validated (train on 2 prior months,
    test on the next). Quantile models (P10/P50/P90) give calibrated uncertainty bands used by the optimizer's
    safety constraint. A separate T_inlet safety model rejects any recommendation predicted to drop tower inlet
    temperature below 300°C. Grate speed and FD fan (coupled to CA) are treated as controllable per the v9 design.
    Dropped features: HAG_Chamber_1_Temp, Chamber_1_Vacuum, Pv_Pa, Y_in_kg_per_kg, Q_CA_m3_per_s,
    mdot_da_CA_kg_per_s, briq_feed_kg_per_hr, W_evap_kg_per_hr, SEC_kJ_per_kg_water (redundant/derived/output metrics).
    </div>
    """, unsafe_allow_html=True)
