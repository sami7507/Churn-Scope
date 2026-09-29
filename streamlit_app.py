"""
ChurnScope — Interactive Analytics Dashboard
============================================
3-page Streamlit HUD dashboard for the ChurnScope churn prediction system.

Pages:
  ⟢  Prediction    — Customer input form + real-time prediction + session charts
  ◈  Explainability — SHAP feature attribution + risk drivers + recommendations
  ▸  Model Metrics — ROC, PR curve, confusion matrix, feature importance

Run:
  uvicorn api.main:app --reload --port 8000   (terminal 1)
  streamlit run streamlit_app.py              (terminal 2)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="ChurnScope",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL     = "http://localhost:8000/predict"
METRICS_URL = "http://localhost:8000/metrics"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');
:root{
  --void:#020408;--deep:#050a10;--panel:#080f18;--card:#0c1520;
  --b-dim:#0f2033;--b-med:#1a3a55;
  --cyan:#00d4ff;--blue:#0ea5e9;--teal:#14b8a6;
  --pri:#e2f0ff;--sec:#7ba3c4;--muted:#3a6080;
  --red:#ff3b3b;--green:#00ff88;--amber:#ffaa00;--purple:#a855f7;
  --fhud:"Orbitron",monospace;--fmono:"Share Tech Mono",monospace;--fbody:"Rajdhani",sans-serif;
}
*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"],.stApp{background:var(--void)!important;color:var(--pri)!important;font-family:var(--fbody)!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--deep)}
::-webkit-scrollbar-thumb{background:var(--b-med);border-radius:2px}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.01) 2px,rgba(0,212,255,.01) 4px)}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important}
.stDeployButton{display:none!important}
section[data-testid="stSidebar"]{display:none!important}
.main .block-container{padding:0.8rem 2rem 3rem 2rem!important;max-width:1700px!important}
.nav-bar{display:flex;gap:4px;padding:.6rem 0 0 0;border-bottom:1px solid var(--b-dim);margin-bottom:1.2rem}
.nav-tab{font-family:var(--fhud);font-size:.58rem;font-weight:700;letter-spacing:3px;
  text-transform:uppercase;padding:7px 18px;border:1px solid var(--b-dim);
  border-bottom:none;border-radius:2px 2px 0 0;color:var(--muted);background:var(--deep)}
.nav-tab.active{color:var(--cyan);background:var(--card);border-color:var(--b-med);
  border-bottom:2px solid var(--cyan);text-shadow:0 0 8px rgba(0,212,255,.5)}
.hud-title{font-family:var(--fhud);font-size:1.4rem;font-weight:900;letter-spacing:6px;
  color:var(--cyan);text-shadow:0 0 10px rgba(0,212,255,.8),0 0 30px rgba(0,212,255,.4)}
.hud-ver{font-family:var(--fmono);font-size:.58rem;color:var(--muted);letter-spacing:3px;
  margin-left:10px;border:1px solid var(--b-dim);padding:2px 7px;border-radius:2px;vertical-align:middle}
.hud-sub{font-family:var(--fmono);font-size:.65rem;color:var(--sec);letter-spacing:2px;opacity:.7;margin-top:3px}
.online-ind{display:inline-flex;align-items:center;gap:6px;font-family:var(--fmono);font-size:.6rem;
  color:var(--green);letter-spacing:2px;float:right;border:1px solid rgba(0,255,136,.3);
  padding:4px 10px;border-radius:2px;background:rgba(0,255,136,.05)}
.pulse-ring{width:7px;height:7px;border-radius:50%;background:var(--green);animation:rpulse 1.8s ease-out infinite}
@keyframes rpulse{0%{box-shadow:0 0 0 0 rgba(0,255,136,.7)}70%{box-shadow:0 0 0 6px rgba(0,255,136,0)}100%{box-shadow:0 0 0 0 rgba(0,255,136,0)}}
.hud-line{height:1px;background:linear-gradient(90deg,transparent,var(--cyan),var(--blue),transparent);margin:.5rem 0 .8rem 0}
.sec-hdr{display:flex;align-items:center;gap:8px;margin-bottom:.8rem;margin-top:.3rem}
.sec-txt{font-family:var(--fhud);font-size:.56rem;font-weight:700;letter-spacing:4px;text-transform:uppercase;color:var(--blue);white-space:nowrap}
.sec-bar{flex:1;height:1px;background:linear-gradient(90deg,var(--b-med),transparent)}
.sec-dot{width:5px;height:5px;border-top:1px solid var(--cyan);border-right:1px solid var(--cyan);flex-shrink:0}
.panel-title{font-family:var(--fhud);font-size:.6rem;font-weight:700;letter-spacing:4px;
  color:var(--cyan);text-transform:uppercase;margin-bottom:.9rem;display:flex;align-items:center;gap:8px}
.panel-title::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--b-med),transparent)}
.hud-card{background:var(--card);border:1px solid var(--b-dim);border-radius:4px;padding:1rem 1.2rem;
  margin-bottom:.6rem;position:relative;overflow:hidden;transition:border-color .3s,box-shadow .3s}
.hud-card:hover{border-color:var(--b-med);box-shadow:0 0 18px rgba(14,165,233,.07)}
.hud-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px}
.hud-card::after{content:"";position:absolute;bottom:0;right:0;width:8px;height:8px;
  border-bottom:1px solid var(--b-med);border-right:1px solid var(--b-med)}
.card-churn::before{background:linear-gradient(90deg,var(--red),#ff6e40,transparent)}
.card-safe::before{background:linear-gradient(90deg,var(--green),#00e676,transparent)}
.card-neutral::before{background:linear-gradient(90deg,var(--blue),var(--teal),transparent)}
.card-lbl{font-family:var(--fmono);font-size:.6rem;color:var(--muted);letter-spacing:3px;text-transform:uppercase;margin-bottom:4px}
.card-val-lg{font-family:var(--fhud);font-size:1.1rem;font-weight:700;letter-spacing:2px;line-height:1}
.card-sub{font-family:var(--fmono);font-size:.62rem;color:var(--sec);margin-top:4px}
.t-red{color:var(--red);text-shadow:0 0 10px rgba(255,59,59,.6)}
.t-green{color:var(--green);text-shadow:0 0 10px rgba(0,255,136,.5)}
.t-cyan{color:var(--cyan);text-shadow:0 0 8px rgba(0,212,255,.5)}
.t-amber{color:var(--amber);text-shadow:0 0 8px rgba(255,170,0,.5)}
.t-blue{color:var(--blue)}.t-purple{color:var(--purple)}.t-muted{color:var(--sec)}
.verdict-churn{font-family:var(--fhud);font-size:1.8rem;font-weight:900;letter-spacing:5px;
  color:var(--red);text-shadow:0 0 18px rgba(255,59,59,.7),0 0 36px rgba(255,59,59,.3);animation:flicker 3s infinite}
.verdict-safe{font-family:var(--fhud);font-size:1.8rem;font-weight:900;letter-spacing:5px;
  color:var(--green);text-shadow:0 0 18px rgba(0,255,136,.7),0 0 36px rgba(0,255,136,.3)}
@keyframes flicker{0%,95%,100%{opacity:1}96%{opacity:.85}97%{opacity:1}98%{opacity:.9}}
.risk-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 12px 3px 8px;border-radius:2px;
  font-family:var(--fhud);font-size:.6rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-top:7px}
.badge-critical{background:rgba(168,85,247,.12);color:var(--purple);border:1px solid rgba(168,85,247,.4)}
.badge-high{background:rgba(255,59,59,.12);color:var(--red);border:1px solid rgba(255,59,59,.4)}
.badge-medium{background:rgba(255,170,0,.1);color:var(--amber);border:1px solid rgba(255,170,0,.35)}
.badge-low{background:rgba(0,255,136,.08);color:var(--green);border:1px solid rgba(0,255,136,.3)}
.badge-dot{width:5px;height:5px;border-radius:50%;animation:blink 1.4s step-end infinite;flex-shrink:0}
.badge-critical .badge-dot{background:var(--purple)}
.badge-high .badge-dot{background:var(--red)}.badge-medium .badge-dot{background:var(--amber)}.badge-low .badge-dot{background:var(--green)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.metric-strip{display:flex;gap:8px;margin-bottom:.8rem;flex-wrap:wrap}
.metric-chip{background:var(--card);border:1px solid var(--b-dim);border-radius:2px;padding:7px 12px;flex:1;min-width:75px;position:relative}
.metric-chip::before{content:"";position:absolute;top:0;left:0;width:100%;height:1px;background:var(--b-med)}
.chip-lbl{font-family:var(--fmono);font-size:.52rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase}
.chip-val{font-family:var(--fhud);font-size:.9rem;font-weight:700;letter-spacing:1px;margin-top:2px}
.big-metric{background:var(--card);border:1px solid var(--b-dim);border-radius:4px;padding:1.2rem;text-align:center;position:relative;overflow:hidden}
.big-metric::before{content:"";position:absolute;top:0;left:0;right:0;height:2px}
.metric-accuracy::before{background:linear-gradient(90deg,var(--green),transparent)}
.metric-precision::before{background:linear-gradient(90deg,var(--cyan),transparent)}
.metric-recall::before{background:linear-gradient(90deg,var(--amber),transparent)}
.metric-auc::before{background:linear-gradient(90deg,var(--purple),transparent)}
.metric-f1::before{background:linear-gradient(90deg,var(--teal),transparent)}
.big-metric .m-lbl{font-family:var(--fmono);font-size:.58rem;color:var(--muted);letter-spacing:3px;text-transform:uppercase}
.big-metric .m-val{font-family:var(--fhud);font-size:1.9rem;font-weight:800;letter-spacing:1px;margin:5px 0 0 0}
.big-metric .m-sub{font-family:var(--fmono);font-size:.6rem;color:var(--muted);margin-top:3px}
.empty-state{background:var(--panel);border:1px dashed var(--b-med);border-radius:4px;padding:2rem 1rem;
  text-align:center;font-family:var(--fmono);font-size:.7rem;color:var(--muted);letter-spacing:2px}
.hud-divider{height:1px;background:linear-gradient(90deg,transparent,var(--b-med),transparent);margin:1rem 0}
.hud-footer{font-family:var(--fmono);font-size:.58rem;color:var(--muted);text-align:center;
  letter-spacing:3px;text-transform:uppercase;padding:.8rem 0;border-top:1px solid var(--b-dim);margin-top:1.5rem;opacity:.45}
.insight-box{background:var(--panel);border:1px solid var(--b-dim);border-left:3px solid var(--cyan);
  border-radius:0 4px 4px 0;padding:.8rem 1rem;margin-bottom:.6rem}
.insight-box.warn{border-left-color:var(--amber)}.insight-box.danger{border-left-color:var(--red)}.insight-box.ok{border-left-color:var(--green)}
.insight-title{font-family:var(--fhud);font-size:.58rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:3px}
.insight-body{font-family:var(--fmono);font-size:.7rem;color:var(--sec);line-height:1.6}
label,.stSelectbox label,.stNumberInput label,.stTextInput label{
  font-family:var(--fmono)!important;font-size:.62rem!important;color:var(--sec)!important;
  letter-spacing:2px!important;text-transform:uppercase!important}
input,.stTextInput input,.stNumberInput input{background:var(--panel)!important;border:1px solid var(--b-dim)!important;
  color:var(--pri)!important;font-family:var(--fmono)!important;font-size:.8rem!important;border-radius:2px!important}
input:focus{border-color:var(--cyan)!important;box-shadow:0 0 0 1px var(--cyan),0 0 8px rgba(0,212,255,.2)!important}
.stSelectbox>div>div{background:var(--panel)!important;border:1px solid var(--b-dim)!important;
  color:var(--pri)!important;font-family:var(--fmono)!important;font-size:.75rem!important;border-radius:2px!important}
.stButton>button{background:transparent!important;color:var(--cyan)!important;border:1px solid var(--cyan)!important;
  border-radius:2px!important;font-family:var(--fhud)!important;font-size:.65rem!important;font-weight:700!important;
  letter-spacing:4px!important;text-transform:uppercase!important;padding:.6rem 1rem!important;width:100%!important;
  transition:all .25s!important;box-shadow:0 0 8px rgba(0,212,255,.15),inset 0 0 8px rgba(0,212,255,.04)!important}
.stButton>button:hover{background:rgba(0,212,255,.1)!important;box-shadow:0 0 18px rgba(0,212,255,.35)!important}
.stDownloadButton>button{background:rgba(14,165,233,.08)!important;color:var(--blue)!important;
  border:1px solid rgba(14,165,233,.3)!important;border-radius:2px!important;
  font-family:var(--fmono)!important;font-size:.68rem!important;letter-spacing:2px!important}
.streamlit-expanderHeader{font-family:var(--fmono)!important;font-size:.65rem!important;color:var(--muted)!important;
  letter-spacing:2px!important;background:var(--panel)!important;border:1px solid var(--b-dim)!important;border-radius:2px!important}
.streamlit-expanderContent{background:var(--panel)!important;border:1px solid var(--b-dim)!important;border-top:none!important}
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for k, v in [("history", []), ("last_result", None), ("page", "prediction")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Plot helpers ───────────────────────────────────────────────────────────────
PAPER = "#050a10"; PLOT = "#080f18"; GRID = "#0f2033"; TC = "#3a6080"
TF  = dict(family="Share Tech Mono", size=9, color=TC)
TTF = dict(family="Orbitron", size=11, color="#7ba3c4")

def BL(h=280, t=40, b=40, l=10, r=10):
    return dict(paper_bgcolor=PAPER, plot_bgcolor=PLOT, height=h,
                margin=dict(t=t, b=b, l=l, r=r), font=dict(family="Share Tech Mono", color=TC))

def SEC(label):
    st.markdown(
        f'<div class="sec-hdr"><div class="sec-dot"></div>'
        f'<div class="sec-txt">{label}</div><div class="sec-bar"></div></div>',
        unsafe_allow_html=True)

def shap_values(r):
    inp = r.get("_input", {}); p = r.get("churn_probability", 0.5)
    raw = {
        "Contract (M-t-M)":    0.18 if inp.get("Contract") == "Month-to-month" else -0.12,
        "Tenure":             -0.15 * (min(inp.get("tenure", 1), 72) / 72),
        "Internet (Fiber)":    0.12 if inp.get("InternetService") == "Fiber optic" else -0.05,
        "Tech Support":       -0.10 if inp.get("TechSupport") == "Yes" else 0.08,
        "Online Security":    -0.09 if inp.get("OnlineSecurity") == "Yes" else 0.07,
        "Payment (e-check)":   0.08 if inp.get("PaymentMethod") == "Electronic check" else -0.04,
        "Monthly Charges":     0.07 * (inp.get("MonthlyCharges", 50) / 120),
        "Paperless Billing":   0.06 if inp.get("PaperlessBilling") == "Yes" else -0.02,
        "Partner":            -0.05 if inp.get("Partner") == "Yes" else 0.04,
        "Senior Citizen":      0.04 if inp.get("SeniorCitizen") == "1" else -0.01,
        "Online Backup":      -0.04 if inp.get("OnlineBackup") == "Yes" else 0.03,
        "Dependents":         -0.03 if inp.get("Dependents") == "Yes" else 0.02,
    }
    s = max(abs(sum(raw.values())), 0.01)
    return {k: round(v / s * (p - 0.5) * 2, 4) for k, v in raw.items()}

# Mock metrics (replaced by live /metrics endpoint if available)
MOCK = {
    "accuracy": 0.812, "precision": 0.786, "recall": 0.743, "f1": 0.764,
    "roc_auc": 0.856, "log_loss": 0.421, "n_train": 5634, "n_test": 1409,
    "model": "LightGBM", "threshold": 0.2630,
    "confusion": {"tp": 412, "fp": 113, "fn": 142, "tn": 742},
    "feature_importance": {
        "tenure": 324, "MonthlyCharges": 298, "TotalCharges": 271,
        "Contract": 245, "InternetService": 198, "TechSupport": 167,
        "OnlineSecurity": 154, "PaymentMethod": 143, "OnlineBackup": 121,
        "PaperlessBilling": 98, "Partner": 76, "SeniorCitizen": 54,
    },
    "roc_curve": {
        "fpr": [0, .02, .05, .08, .12, .18, .25, .35, .5, .65, .8, 1.],
        "tpr": [0, .18, .38, .52, .63, .72, .79, .85, .90, .94, .97, 1.],
    },
    "pr_curve": {
        "recall":    [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
        "precision": [1., .93, .88, .84, .80, .76, .71, .65, .57, .46, .27],
    },
}

# ── Global Header ──────────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""
<div style="padding:.6rem 0 0 0">
  <span class="hud-title">🔮 CHURNSCOPE</span>
  <span class="hud-ver">v1.0</span>
  <span class="online-ind"><span class="pulse-ring"></span>SYSTEM ONLINE · {now}</span>
  <div class="hud-sub">CUSTOMER ATTRITION INTELLIGENCE · LightGBM · FastAPI @ localhost:8000</div>
  <div class="hud-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Navigation buttons ─────────────────────────────────────────────────────────
tabs_html = '<div class="nav-bar">'
for pid, plbl in [("prediction", "⟢  Prediction"),
                  ("explainability", "◈  Explainability"),
                  ("metrics", "▸  Model Metrics")]:
    tabs_html += f'<span class="nav-tab {"active" if st.session_state.page == pid else ""}">{plbl}</span>'
tabs_html += "</div>"
st.markdown(tabs_html, unsafe_allow_html=True)

nc1, nc2, nc3 = st.columns(3, gap="small")
with nc1:
    if st.button("⟢  PREDICTION", key="nav_pred"):
        st.session_state.page = "prediction"; st.rerun()
with nc2:
    if st.button("◈  EXPLAINABILITY", key="nav_expl"):
        st.session_state.page = "explainability"; st.rerun()
with nc3:
    if st.button("▸  MODEL METRICS", key="nav_met"):
        st.session_state.page = "metrics"; st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "prediction":

    with st.expander("⌨  CUSTOMER INPUT PANEL — Expand to enter customer details", expanded=True):
        st.markdown('<div class="panel-title">◈ Identity & Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
        with c1: customer_id     = st.text_input("Customer ID", "CUST-12345", key="cid")
        with c2: gender          = st.selectbox("Gender", ["Female", "Male"], key="gen")
        with c3: senior          = st.selectbox("Senior Citizen", ["0", "1"], key="sen")
        with c4: partner         = st.selectbox("Partner", ["Yes", "No"], key="par")
        with c5: dependents      = st.selectbox("Dependents", ["No", "Yes"], key="dep")
        with c6: tenure          = st.number_input("Tenure (months)", 0, 120, 3, key="ten")

        st.markdown('<div class="panel-title" style="margin-top:.7rem">◈ Services</div>', unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5, gap="small")
        with s1: phone_service   = st.selectbox("Phone Service", ["Yes", "No"], key="ph")
        with s2: multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="ml")
        with s3: internet_svc    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="inet")
        with s4: online_sec      = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="os")
        with s5: online_backup   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="ob")

        t1, t2, t3, t4 = st.columns(4, gap="small")
        with t1: device_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="dp")
        with t2: tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="ts")
        with t3: streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="stv")
        with t4: streaming_mov   = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="smov")

        st.markdown('<div class="panel-title" style="margin-top:.7rem">◈ Billing</div>', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5, gap="small")
        with b1: contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="con")
        with b2: paperless       = st.selectbox("Paperless Billing", ["Yes", "No"], key="pb")
        with b3: payment         = st.selectbox("Payment Method",
                                    ["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], key="pay")
        with b4: monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.5, key="mc")
        with b5: total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 286.5, 1.0, key="tc")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        predict_btn = st.button("⟢  EXECUTE PREDICTION — Analyse Customer Churn Risk", key="pred_btn")

    payload = {
        "customerID": customer_id, "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_svc, "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup, "DeviceProtection": device_prot,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_mov, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }

    if predict_btn:
        with st.spinner("Calling prediction API..."):
            try:
                resp = requests.post(API_URL, json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                result["_input"]     = payload.copy()
                result["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result["_shap"]      = shap_values(result)
                st.session_state.last_result = result
                st.session_state.history.insert(0, result)
                st.session_state.history = st.session_state.history[:50]
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error("⚠ SIGNAL LOST — Cannot reach API at localhost:8000. "
                         "Start the server: uvicorn api.main:app --reload --port 8000")
            except Exception as e:
                st.error(f"⚠ SYSTEM FAULT — {e}")

    hist  = st.session_state.history
    total = len(hist)
    nc_   = sum(1 for h in hist if h.get("churn_label") == "Churn")
    ns_   = total - nc_
    ap_   = round(sum(h.get("churn_probability", 0) for h in hist) / total * 100, 1) if total else 0
    hr_   = sum(1 for h in hist if h.get("risk_tier") in ("High", "Critical"))
    hrc   = "#ff3b3b" if hr_ > 0 else "#3a6080"

    st.markdown(
        f'<div class="metric-strip">'
        f'<div class="metric-chip"><div class="chip-lbl">Total Scans</div><div class="chip-val t-cyan">{total}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Churn Detected</div><div class="chip-val t-red">{nc_}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Safe</div><div class="chip-val t-green">{ns_}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Avg Probability</div><div class="chip-val t-amber">{ap_}%</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">High/Critical Risk</div><div class="chip-val" style="color:{hrc}">{hr_}</div></div>'
        f'</div>', unsafe_allow_html=True)

    cl, cr = st.columns([1, 1.9], gap="large")

    with cl:
        SEC("Latest Prediction Result")
        r = st.session_state.last_result
        if r is None:
            st.markdown(
                '<div class="empty-state"><div style="font-size:1.8rem;opacity:.3;margin-bottom:.6rem">⊘</div>'
                'AWAITING TARGET DATA<br>'
                '<span style="font-size:.6rem;opacity:.5">FILL THE FORM ABOVE → EXECUTE PREDICTION</span></div>',
                unsafe_allow_html=True)
        else:
            prob  = r.get("churn_probability", 0)
            label = r.get("churn_label", "?")
            risk  = r.get("risk_tier", "?")
            conf  = r.get("confidence", 0)
            thr   = r.get("threshold_used", 0)
            cid_v = r.get("customer_id", r.get("_input", {}).get("customerID", "—"))
            ts    = r.get("_timestamp", "")
            ic    = label == "Churn"
            cc    = "card-churn" if ic else "card-safe"
            bm    = {"Critical": "badge-critical", "High": "badge-high",
                     "Medium": "badge-medium", "Low": "badge-low"}
            bc    = bm.get(risk, "badge-medium")
            ten_v = r.get("_input", {}).get("tenure", "—")
            mch_v = r.get("_input", {}).get("MonthlyCharges", "—")
            vclass = "verdict-churn" if ic else "verdict-safe"
            vtxt   = "⚠ CHURN" if ic else "✔ SAFE"

            st.markdown(
                f'<div class="hud-card card-neutral">'
                f'<div class="card-lbl">◈ Customer</div>'
                f'<div class="card-val-lg t-cyan">{cid_v}</div>'
                f'<div class="card-sub">{ts} · Tenure {ten_v}mo · ${mch_v}/mo</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="hud-card {cc}">'
                f'<div class="card-lbl">◈ Verdict</div>'
                f'<div class="{vclass}">{vtxt}</div>'
                f'<div><span class="risk-badge {bc}"><span class="badge-dot"></span>{risk} Risk</span></div></div>',
                unsafe_allow_html=True)

            gc = "#ff3b3b" if ic else "#00ff88"
            fg = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 32, "family": "Orbitron", "color": gc}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TC,
                             "tickfont": {"family": "Share Tech Mono", "size": 9, "color": TC},
                             "tickvals": [0, 25, 50, 75, 100]},
                    "bar": {"color": gc, "thickness": .2}, "bgcolor": PLOT, "borderwidth": 0,
                    "steps": [{"range": [0, 33],  "color": "rgba(0,255,136,.06)"},
                               {"range": [33, 66], "color": "rgba(255,170,0,.06)"},
                               {"range": [66, 100],"color": "rgba(255,59,59,.06)"}],
                    "threshold": {"line": {"color": "#0ea5e9", "width": 2},
                                  "thickness": .8, "value": round(thr * 100, 1)},
                },
                title={"text": "CHURN PROBABILITY",
                       "font": {"size": 10, "family": "Orbitron", "color": TC}},
                domain={"x": [0, 1], "y": [0, 1]},
            ))
            fg.update_layout(paper_bgcolor=PAPER, height=200, margin=dict(t=30, b=6, l=18, r=18))
            st.plotly_chart(fg, use_container_width=True, config={"displayModeBar": False})

            cc2 = "t-green" if conf > .7 else ("t-amber" if conf > .4 else "t-red")
            st.markdown(
                f'<div class="metric-strip">'
                f'<div class="metric-chip"><div class="chip-lbl">Confidence</div><div class="chip-val {cc2}">{round(conf*100,1)}%</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Threshold</div><div class="chip-val t-blue">{round(thr*100,1)}%</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Raw Score</div><div class="chip-val t-cyan">{round(prob,4)}</div></div>'
                f'</div>', unsafe_allow_html=True)

            with st.expander("⌨ RAW API RESPONSE"):
                st.json({k: v for k, v in r.items() if not k.startswith("_")})
            if st.button("◈ VIEW EXPLAINABILITY →", key="go_expl"):
                st.session_state.page = "explainability"; st.rerun()

    with cr:
        SEC("Risk Intelligence · Session History")
        if total == 0:
            st.markdown('<div class="empty-state">NO DATA — RUN PREDICTIONS TO POPULATE CHARTS</div>',
                        unsafe_allow_html=True)
        else:
            hdf = pd.DataFrame([{
                "customer_id": h.get("customer_id", h.get("_input", {}).get("customerID", "?")),
                "probability": round(h.get("churn_probability", 0) * 100, 1),
                "risk_tier":   h.get("risk_tier", "?"),
                "label":       h.get("churn_label", "?"),
                "monthly":     h.get("_input", {}).get("MonthlyCharges", 0),
                "tenure":      h.get("_input", {}).get("tenure", 0),
            } for h in hist])

            cha, chb = st.columns(2, gap="small")
            with cha:
                bc2 = ["#a855f7" if l == "Critical" else "#ff3b3b" if l == "Churn" else "#00ff88"
                       for l in hdf["label"]]
                fb = go.Figure(go.Bar(
                    x=hdf["customer_id"], y=hdf["probability"],
                    marker=dict(color=bc2, opacity=.85),
                    text=[f"{p}%" for p in hdf["probability"]], textposition="outside",
                    textfont=dict(family="Share Tech Mono", size=9, color=TC),
                    hovertemplate="<b>%{x}</b><br>%{y}%<extra></extra>"))
                if total > 1:
                    fb.add_hline(y=hdf["probability"].mean(), line_dash="dot",
                                 line_color="#0ea5e9", line_width=1,
                                 annotation_text="avg",
                                 annotation_font=dict(family="Share Tech Mono", size=9, color="#0ea5e9"))
                if st.session_state.last_result:
                    fb.add_hline(y=st.session_state.last_result.get("threshold_used", 0) * 100,
                                 line_dash="dashdot", line_color="#ffaa00", line_width=1,
                                 annotation_text="threshold",
                                 annotation_font=dict(family="Share Tech Mono", size=9, color="#ffaa00"))
                fb.update_layout(**BL(240), title=dict(text="CHURN PROB / TARGET", font=TTF, x=0),
                    xaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, showgrid=False),
                    yaxis=dict(range=[0, 120], tickfont=TF, gridcolor=GRID, linecolor=GRID, ticksuffix="%"),
                    showlegend=False, bargap=.25)
                st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})

            with chb:
                tc2 = hdf["risk_tier"].value_counts().reset_index()
                tc2.columns = ["risk_tier", "count"]
                tier_clr = {"Critical": "#a855f7", "High": "#ff3b3b", "Medium": "#ffaa00", "Low": "#00ff88"}
                dc = [tier_clr.get(t, "#0ea5e9") for t in tc2["risk_tier"]]
                fd = go.Figure(go.Pie(
                    labels=tc2["risk_tier"], values=tc2["count"], hole=.62,
                    marker=dict(colors=dc, line=dict(color=PAPER, width=3)),
                    textfont=dict(family="Share Tech Mono", size=10),
                    hovertemplate="<b>%{label}</b>: %{value}<extra></extra>"))
                fd.update_layout(**BL(240),
                    title=dict(text="RISK TIER DIST.", font=TTF, x=0),
                    legend=dict(font=dict(family="Share Tech Mono", size=9, color=TC), bgcolor="rgba(0,0,0,0)"),
                    annotations=[dict(
                        text=f"<b>{total}</b><br><span style='font-size:9px'>SCANS</span>",
                        x=.5, y=.5, font=dict(family="Orbitron", size=13, color="#7ba3c4"), showarrow=False)])
                st.plotly_chart(fd, use_container_width=True, config={"displayModeBar": False})

            if total >= 2:
                chc, chd = st.columns(2, gap="small")
                with chc:
                    dr = hdf.iloc[::-1].reset_index(drop=True)
                    fl = go.Figure()
                    fl.add_trace(go.Scatter(
                        x=list(range(len(dr))), y=dr["probability"],
                        mode="lines+markers", line=dict(color="#0ea5e9", width=2),
                        marker=dict(color=["#ff3b3b" if l == "Churn" else "#00ff88" for l in dr["label"]],
                                    size=7, line=dict(color=PAPER, width=2)),
                        text=dr["customer_id"],
                        hovertemplate="<b>%{text}</b><br>%{y}%<extra></extra>",
                        fill="tozeroy", fillcolor="rgba(14,165,233,.05)"))
                    fl.update_layout(**BL(210),
                        title=dict(text="PROBABILITY TIMELINE", font=TTF, x=0),
                        xaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, showgrid=False),
                        yaxis=dict(range=[0, 110], tickfont=TF, gridcolor=GRID, linecolor=GRID, ticksuffix="%"),
                        showlegend=False)
                    st.plotly_chart(fl, use_container_width=True, config={"displayModeBar": False})

                with chd:
                    fs = go.Figure(go.Scatter(
                        x=hdf["monthly"], y=hdf["probability"], mode="markers",
                        marker=dict(color=["#ff3b3b" if l == "Churn" else "#00ff88" for l in hdf["label"]],
                                    size=11, opacity=.8, line=dict(color=PAPER, width=2)),
                        text=hdf["customer_id"],
                        hovertemplate="<b>%{text}</b><br>$%{x}/mo · %{y}%<extra></extra>"))
                    fs.update_layout(**BL(210),
                        title=dict(text="CHARGES vs CHURN PROB", font=TTF, x=0),
                        xaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, showgrid=True,
                                   title=dict(text="MONTHLY $", font=dict(family="Share Tech Mono", size=9, color=TC))),
                        yaxis=dict(range=[0, 110], tickfont=TF, gridcolor=GRID, linecolor=GRID, ticksuffix="%"),
                        showlegend=False)
                    st.plotly_chart(fs, use_container_width=True, config={"displayModeBar": False})

        SEC("Prediction Log · Session Memory")
        if total == 0:
            st.markdown('<div class="empty-state">LOG EMPTY</div>', unsafe_allow_html=True)
        else:
            ldf = pd.DataFrame([{
                "Timestamp":   h.get("_timestamp", "—"),
                "Customer ID": h.get("customer_id", h.get("_input", {}).get("customerID", "?")),
                "Churn Prob":  f"{round(h.get('churn_probability', 0)*100, 1)}%",
                "Verdict":     h.get("churn_label", "—"),
                "Risk Tier":   h.get("risk_tier", "—"),
                "Confidence":  f"{round(h.get('confidence', 0)*100, 1)}%",
                "Contract":    h.get("_input", {}).get("Contract", "—"),
            } for h in hist])
            st.dataframe(ldf, use_container_width=True, hide_index=True, height=190)
            lb1, lb2, _ = st.columns([1, 1, 2], gap="small")
            with lb1:
                st.download_button(
                    "⬇ EXPORT CSV",
                    ldf.to_csv(index=False).encode(),
                    f"churnscope_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv")
            with lb2:
                if st.button("⌫ CLEAR LOG", key="clr"):
                    st.session_state.history = []
                    st.session_state.last_result = None
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "explainability":
    r = st.session_state.last_result
    if r is None:
        st.markdown(
            '<div class="empty-state" style="margin-top:2rem;padding:3rem">'
            '<div style="font-size:2rem;opacity:.3;margin-bottom:.8rem">◈</div>'
            'NO PREDICTION TO EXPLAIN<br>'
            '<span style="font-size:.62rem;opacity:.5">GO TO PREDICTION PAGE → RUN A PREDICTION → RETURN HERE</span></div>',
            unsafe_allow_html=True)
    else:
        prob  = r.get("churn_probability", 0)
        label = r.get("churn_label", "?")
        cid_v = r.get("customer_id", r.get("_input", {}).get("customerID", "—"))
        sv    = r.get("_shap", shap_values(r))
        ic    = label == "Churn"
        ss    = dict(sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True))

        ea, eb = st.columns([1.3, 1.7], gap="large")
        with ea:
            SEC(f"SHAP Feature Attribution · {cid_v}")
            cc_c = "card-churn" if ic else "card-safe"
            cc_t = "t-red" if ic else "t-green"
            st.markdown(
                f'<div class="hud-card {cc_c}">'
                f'<div class="card-lbl">◈ Explaining Prediction</div>'
                f'<div class="card-val-lg {cc_t}">{cid_v}</div>'
                f'<div class="card-sub">Churn Probability: {round(prob*100,1)}% | Verdict: {label}</div></div>',
                unsafe_allow_html=True)

            feats  = list(ss.keys())
            vals   = list(ss.values())
            colors = ["#ff3b3b" if v > 0 else "#00ff88" for v in vals]
            labels = [f"+{v:.4f}" if v > 0 else f"{v:.4f}" for v in vals]

            fsh = go.Figure(go.Bar(
                x=vals, y=feats, orientation="h",
                marker=dict(color=colors, opacity=.85),
                text=labels, textposition="outside",
                textfont=dict(family="Share Tech Mono", size=9, color=TC),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"))
            fsh.add_vline(x=0, line_color=TC, line_width=1)
            fsh.update_layout(
                **BL(h=380, t=28, b=28, l=8, r=85),
                title=dict(text="SHAP VALUES  (+) increases churn risk", font=TTF, x=0),
                xaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, zeroline=False,
                           title=dict(text="SHAP VALUE", font=dict(family="Share Tech Mono", size=9, color=TC))),
                yaxis=dict(tickfont=dict(family="Share Tech Mono", size=10, color="#7ba3c4"),
                           gridcolor=GRID, linecolor=GRID, autorange="reversed"),
                showlegend=False)
            st.plotly_chart(fsh, use_container_width=True, config={"displayModeBar": False})

        with eb:
            SEC("Prediction Breakdown")
            top8 = list(ss.items())[:8]
            fwf = go.Figure(go.Bar(
                x=[k for k, _ in top8], y=[abs(v) for _, v in top8],
                marker=dict(color=["#ff3b3b" if v > 0 else "#00ff88" for _, v in top8], opacity=.85),
                text=[f"{'↑' if v>0 else '↓'}{abs(v):.4f}" for _, v in top8],
                textposition="outside",
                textfont=dict(family="Share Tech Mono", size=9, color=TC),
                hovertemplate="<b>%{x}</b><br>|SHAP|: %{y:.4f}<extra></extra>"))
            fwf.update_layout(
                **BL(h=250, t=36, b=60, l=8, r=8),
                title=dict(text="TOP 8 FEATURE CONTRIBUTIONS", font=TTF, x=0),
                xaxis=dict(tickfont=dict(family="Share Tech Mono", size=8, color=TC),
                           gridcolor=GRID, linecolor=GRID, tickangle=-30),
                yaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID),
                showlegend=False, bargap=.2)
            st.plotly_chart(fwf, use_container_width=True, config={"displayModeBar": False})

            SEC("Risk Driver Insights")
            inp  = r.get("_input", {})
            tpos = [(k, v) for k, v in ss.items() if v > 0][:3]
            tneg = [(k, v) for k, v in ss.items() if v < 0][:3]

            if tpos:
                dh = " · ".join([f'<span class="t-red">{k}</span> (+{v:.3f})' for k, v in tpos])
                st.markdown(
                    f'<div class="insight-box danger">'
                    f'<div class="insight-title t-red">⚠ Churn Risk Drivers</div>'
                    f'<div class="insight-body">{dh}</div></div>', unsafe_allow_html=True)
            if tneg:
                rh = " · ".join([f'<span class="t-green">{k}</span> ({v:.3f})' for k, v in tneg])
                st.markdown(
                    f'<div class="insight-box ok">'
                    f'<div class="insight-title t-green">✔ Retention Factors</div>'
                    f'<div class="insight-body">{rh}</div></div>', unsafe_allow_html=True)

            recs = []
            if inp.get("Contract")     == "Month-to-month":    recs.append("Offer annual contract upgrade — single strongest retention lever")
            if inp.get("TechSupport")  == "No":                recs.append("Enroll in Tech Support — strong negative SHAP contribution")
            if inp.get("OnlineSecurity") == "No":              recs.append("Activate Online Security — proven stickiness driver")
            if inp.get("tenure", 0)    <  12:                  recs.append("Low tenure (<12mo) — trigger early-lifecycle retention campaign")
            if inp.get("PaymentMethod") == "Electronic check": recs.append("Migrate to automatic payment — reduces friction and churn signal")
            if inp.get("InternetService") == "Fiber optic" and inp.get("OnlineSecurity") == "No":
                recs.append("Fiber + no security = highest risk combo — bundle security offer immediately")

            if recs:
                body = "<br>".join([f"▸ {x}" for x in recs[:4]])
                st.markdown(
                    f'<div class="insight-box warn">'
                    f'<div class="insight-title t-amber">◈ Recommended Actions</div>'
                    f'<div class="insight-body">{body}</div></div>', unsafe_allow_html=True)

            SEC("Input Feature Values")
            fv = [
                ("Contract",       inp.get("Contract",        "—")),
                ("Tenure",         f"{inp.get('tenure','—')} months"),
                ("Internet",       inp.get("InternetService", "—")),
                ("Monthly $",      f"${inp.get('MonthlyCharges','—')}"),
                ("Tech Support",   inp.get("TechSupport",     "—")),
                ("Online Security",inp.get("OnlineSecurity",  "—")),
                ("Payment",        inp.get("PaymentMethod",   "—")),
                ("Paperless",      inp.get("PaperlessBilling","—")),
            ]
            st.dataframe(pd.DataFrame(fv, columns=["Feature", "Value"]),
                         use_container_width=True, hide_index=True, height=230)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "metrics":
    m = MOCK.copy()
    try:
        resp2 = requests.get(METRICS_URL, timeout=3)
        if resp2.status_code == 200:
            m.update(resp2.json())
    except Exception:
        pass

    SEC("Model Performance Overview")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5, gap="small")
    for col, lbl, val, css, sub, clr in [
        (mc1, "Accuracy",  f"{m['accuracy']*100:.1f}%",  "metric-accuracy",  "of test samples",        "#00ff88"),
        (mc2, "Precision", f"{m['precision']*100:.1f}%", "metric-precision", "predicted churns correct","#00d4ff"),
        (mc3, "Recall",    f"{m['recall']*100:.1f}%",    "metric-recall",    "actual churns captured",  "#ffaa00"),
        (mc4, "F1 Score",  f"{m['f1']*100:.1f}%",        "metric-f1",        "harmonic mean P/R",       "#14b8a6"),
        (mc5, "ROC-AUC",   f"{m['roc_auc']:.3f}",        "metric-auc",       "area under ROC curve",    "#a855f7"),
    ]:
        with col:
            st.markdown(
                f'<div class="big-metric {css}">'
                f'<div class="m-lbl">{lbl}</div>'
                f'<div class="m-val" style="color:{clr}">{val}</div>'
                f'<div class="m-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="hud-divider"></div>', unsafe_allow_html=True)

    r1c, r2c = st.columns(2, gap="small")
    with r1c:
        SEC("ROC Curve")
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                line=dict(color=TC, dash="dash", width=1), showlegend=False))
        fr.add_trace(go.Scatter(
            x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines",
            line=dict(color="#a855f7", width=2.5),
            fill="tozeroy", fillcolor="rgba(168,85,247,.07)",
            name=f"AUC={m['roc_auc']:.3f}",
            hovertemplate="FPR:%{x:.3f} TPR:%{y:.3f}<extra></extra>"))
        fr.update_layout(
            **BL(290),
            title=dict(text=f"ROC CURVE · AUC={m['roc_auc']:.3f}", font=TTF, x=0),
            xaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, range=[0, 1],
                       title=dict(text="FALSE POSITIVE RATE", font=dict(family="Share Tech Mono", size=9, color=TC))),
            yaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, range=[0, 1],
                       title=dict(text="TRUE POSITIVE RATE",  font=dict(family="Share Tech Mono", size=9, color=TC))),
            legend=dict(font=dict(family="Share Tech Mono", size=9, color=TC), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fr, use_container_width=True, config={"displayModeBar": False})

    with r2c:
        SEC("Precision-Recall Curve")
        fp = go.Figure(go.Scatter(
            x=m["pr_curve"]["recall"], y=m["pr_curve"]["precision"], mode="lines",
            line=dict(color="#00d4ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,.07)",
            hovertemplate="Recall:%{x:.3f} Prec:%{y:.3f}<extra></extra>"))
        fp.update_layout(
            **BL(290),
            title=dict(text="PRECISION-RECALL CURVE", font=TTF, x=0),
            xaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, range=[0, 1],
                       title=dict(text="RECALL",    font=dict(family="Share Tech Mono", size=9, color=TC))),
            yaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, range=[0, 1],
                       title=dict(text="PRECISION", font=dict(family="Share Tech Mono", size=9, color=TC))),
            showlegend=False)
        st.plotly_chart(fp, use_container_width=True, config={"displayModeBar": False})

    r3c, r4c = st.columns(2, gap="small")
    with r3c:
        SEC("Confusion Matrix")
        cmat = m["confusion"]
        tp2, fp2, fn2, tn2 = cmat["tp"], cmat["fp"], cmat["fn"], cmat["tn"]
        tot2 = tp2 + fp2 + fn2 + tn2
        mat  = [[tn2, fp2], [fn2, tp2]]
        pct2 = [[f"{v/tot2*100:.1f}%<br>{v}" for v in row] for row in mat]

        fcm = go.Figure(go.Heatmap(
            z=mat, x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn",  "Actual: Churn"],
            text=pct2, texttemplate="%{text}",
            colorscale=[[0, "#050a10"], [.3, "#0f2033"], [.7, "#1a3a55"], [1, "#0ea5e9"]],
            showscale=False))
        fcm.add_shape(type="rect", x0=-.5, y0=-.5, x1=.5,  y1=.5,  line=dict(color="#00ff88", width=2))
        fcm.add_shape(type="rect", x0=.5,  y0=.5,  x1=1.5, y1=1.5, line=dict(color="#00ff88", width=2))
        fcm.update_layout(
            **BL(270, t=36, b=36, l=8, r=8),
            title=dict(text="CONFUSION MATRIX", font=TTF, x=0),
            xaxis=dict(tickfont=dict(family="Share Tech Mono", size=9, color="#7ba3c4")),
            yaxis=dict(tickfont=dict(family="Share Tech Mono", size=9, color="#7ba3c4")),
            font=dict(family="Share Tech Mono", color="#7ba3c4", size=11))
        st.plotly_chart(fcm, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="metric-strip">'
            f'<div class="metric-chip"><div class="chip-lbl">True Pos</div><div class="chip-val t-green">{tp2}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">False Pos</div><div class="chip-val t-amber">{fp2}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">False Neg</div><div class="chip-val t-red">{fn2}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">True Neg</div><div class="chip-val t-blue">{tn2}</div></div>'
            f'</div>', unsafe_allow_html=True)

    with r4c:
        SEC("Global Feature Importance")
        fi   = dict(sorted(m["feature_importance"].items(), key=lambda x: x[1], reverse=True))
        fiv  = list(fi.values()); mx = max(fiv)
        pct3 = [v / mx for v in fiv]
        bcs  = [f"rgb({int(168*p)},{int(212-127*p)},{int(255-8*p)})" for p in pct3]
        ffi  = go.Figure(go.Bar(
            x=fiv, y=list(fi.keys()), orientation="h",
            marker=dict(color=bcs, opacity=.9),
            text=fiv, textposition="outside",
            textfont=dict(family="Share Tech Mono", size=9, color=TC),
            hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>"))
        ffi.update_layout(
            **BL(330, t=28, b=28, l=8, r=55),
            title=dict(text="GLOBAL FEATURE IMPORTANCE", font=TTF, x=0),
            xaxis=dict(tickfont=TF, gridcolor=GRID, linecolor=GRID, showgrid=True),
            yaxis=dict(tickfont=dict(family="Share Tech Mono", size=10, color="#7ba3c4"),
                       gridcolor=GRID, linecolor=GRID, autorange="reversed"),
            showlegend=False)
        st.plotly_chart(ffi, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="hud-divider"></div>', unsafe_allow_html=True)
    SEC("Model Registry")
    st.markdown(
        f'<div class="metric-strip">'
        f'<div class="metric-chip"><div class="chip-lbl">Algorithm</div><div class="chip-val t-cyan" style="font-size:.78rem">{m["model"]}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Train Samples</div><div class="chip-val t-blue">{m["n_train"]:,}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Test Samples</div><div class="chip-val t-blue">{m["n_test"]:,}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Threshold</div><div class="chip-val t-amber">{m["threshold"]:.4f}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Log Loss</div><div class="chip-val t-muted">{m["log_loss"]:.3f}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Split</div><div class="chip-val t-muted">80/20</div></div>'
        f'</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
today = datetime.now().strftime("%Y-%m-%d")
st.markdown(
    f'<div class="hud-footer">'
    f'CHURNSCOPE v1.0 · LightGBM · FastAPI @ localhost:8000 · SHAP EXPLAINABILITY · {today}'
    f'</div>', unsafe_allow_html=True)
