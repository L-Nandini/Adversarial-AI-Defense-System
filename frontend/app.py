# frontend/app.py
import streamlit as st
import requests
import pandas as pd
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Vigil Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"

# =========================
# GLOBAL STYLES (Modern UI)
# =========================
st.markdown(
    """
<style>
/* -------- App background -------- */
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(124,58,237,0.25), transparent 60%),
              radial-gradient(900px 500px at 90% 20%, rgba(59,130,246,0.22), transparent 55%),
              radial-gradient(1000px 600px at 30% 90%, rgba(16,185,129,0.18), transparent 60%),
              linear-gradient(135deg, #070A12 0%, #0B1220 45%, #070A12 100%);
}

/* -------- Hide Streamlit default header/footer -------- */
/* header {visibility: hidden;} */
footer {visibility: hidden;}

/* -------- Sidebar styling -------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(17,24,39,0.95) 0%, rgba(10,16,28,0.95) 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] label, 
section[data-testid="stSidebar"] p {
  color: rgba(255,255,255,0.9) !important;
}

/* -------- Buttons -------- */
.stButton button {
  width: 100%;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: linear-gradient(135deg, rgba(124,58,237,0.85), rgba(59,130,246,0.85)) !important;
  color: #fff !important;
  border-radius: 14px !important;
  padding: 0.75rem 1rem !important;
  font-weight: 700 !important;
  transition: transform 0.08s ease, box-shadow 0.2s ease !important;
  box-shadow: 0 10px 24px rgba(0,0,0,0.25) !important;
}
.stButton button:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(0,0,0,0.35) !important;
}
.stButton button:active { transform: translateY(0px); }

/* -------- Cards -------- */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 18px 45px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}

/* -------- Title block -------- */
.hero {
  border-radius: 22px;
  padding: 20px 22px;
  background:
    radial-gradient(900px 200px at 10% 0%, rgba(124,58,237,0.35), transparent 60%),
    radial-gradient(900px 250px at 90% 0%, rgba(59,130,246,0.32), transparent 60%),
    rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 18px 50px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}
.hero h1 {
  margin: 0;
  font-size: 2.0rem;
  letter-spacing: 0.2px;
  color: rgba(255,255,255,0.95);
}
.hero p {
  margin: 6px 0 0 0;
  color: rgba(255,255,255,0.75);
  font-size: 0.98rem;
}

/* -------- KPI pill -------- */
.kpi {
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
}
.kpi .label { color: rgba(255,255,255,0.70); font-size: 0.85rem; }
.kpi .value { color: rgba(255,255,255,0.95); font-weight: 800; font-size: 1.25rem; }

/* -------- Status banners -------- */
.banner-ok {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(16,185,129,0.12);
  border: 1px solid rgba(16,185,129,0.25);
  color: rgba(255,255,255,0.92);
}
.banner-bad {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(239,68,68,0.10);
  border: 1px solid rgba(239,68,68,0.25);
  color: rgba(255,255,255,0.92);
}
.banner-warn {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(245,158,11,0.10);
  border: 1px solid rgba(245,158,11,0.25);
  color: rgba(255,255,255,0.92);
}

/* -------- Dataframe style tweaks -------- */
div[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
}

/* -------- Upload box -------- */
div[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.03);
  border: 1px dashed rgba(255,255,255,0.18);
  border-radius: 18px;
  padding: 10px 10px;
}

/* -------- Small text -------- */
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.90rem; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 16px 0; }
</style>
    """,
    unsafe_allow_html=True
)

# =========================
# HELPERS
# =========================
def ping_backend() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def pretty_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 🛡️ AI Vigil Guard")
st.sidebar.markdown('<p class="small-muted">Real-time adversarial detection for CSV & Images</p>', unsafe_allow_html=True)

pages = {
    "CSV Detection": "📊 CSV Adversarial Detection",
    "Image Detection": "🖼️ Image Adversarial Detection",
    "Operator Panel": "⚙️ Operator Control Panel",
}
page_key = st.sidebar.radio("Navigation", list(pages.keys()))
page = pages[page_key]

backend_live = ping_backend()
if backend_live:
    st.sidebar.success("Backend: Connected ✅")
else:
    st.sidebar.error("Backend: Offline ❌")
    st.sidebar.caption("Start backend: `python -m uvicorn main:app --reload`")

st.sidebar.markdown("---")
st.sidebar.markdown("### UI Options")
show_technical = st.sidebar.toggle("Show technical metrics", value=True)
compact_mode = st.sidebar.toggle("Compact layout", value=False)

# =========================
# HERO HEADER
# =========================
st.markdown(
    """
<div class="hero">
  <h1>🛡️ AI Vigil Guard</h1>
  <p>Adversarial Attack Detection • Explainable signals • FastAPI + Streamlit</p>
</div>
""",
    unsafe_allow_html=True
)

# KPI row
k1, k2, k3, k4 = st.columns([1.1,1,1,1])
with k1:
    st.markdown(f"""
    <div class="kpi">
      <div class="label">System Status</div>
      <div class="value">{'Online ✅' if backend_live else 'Offline ❌'}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown("""
    <div class="kpi">
      <div class="label">Mode</div>
      <div class="value">Detection</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown("""
    <div class="kpi">
      <div class="label">Explainability</div>
      <div class="value">Enabled</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown("""
    <div class="kpi">
      <div class="label">Pipeline</div>
      <div class="value">Upload → Analyze</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# CSV PAGE
# =========================
if page == "📊 CSV Adversarial Detection":
    left, right = st.columns([1.15, 0.85] if not compact_mode else [1,1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 CSV Adversarial Detection")
        st.markdown('<p class="small-muted">Upload a CSV and detect suspicious rows (demo logic from backend).</p>', unsafe_allow_html=True)

        csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")

        if csv_file:
            df = pd.read_csv(csv_file)
            st.write("### Preview")
            st.dataframe(df.head(15), use_container_width=True)

            st.caption(f"File: **{csv_file.name}** • Size: **{pretty_bytes(len(csv_file.getvalue()))}** • Rows: **{len(df)}**")

            if st.button("🔍 Run CSV Detection"):
                if not backend_live:
                    st.error("Backend is offline. Please start FastAPI backend first.")
                else:
                    with st.spinner("Analyzing CSV..."):
                        files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
                        r = requests.post(f"{BACKEND_URL}/detect/csv", files=files, timeout=60)

                    if r.status_code == 200:
                        out = r.json()
                        st.success("Detection Complete ✅")
                        st.json(out)
                    else:
                        st.error(r.text)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 What it checks")
        st.markdown("""
- File format validation  
- Basic dataset size checks  
- Returns a suspicious ratio output (demo)  
        """)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("✅ Tips")
        st.markdown("""
- Use clean numeric columns for best demo results  
- If you change backend routes, update `BACKEND_URL`  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# IMAGE PAGE
# =========================
elif page == "🖼️ Image Adversarial Detection":
    left, right = st.columns([1.2, 0.8] if not compact_mode else [1,1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼️ Image Adversarial Detection")
        st.markdown('<p class="small-muted">Upload an image and detect whether it looks adversarial. The backend returns an explainable reason list.</p>', unsafe_allow_html=True)

        image_file = st.file_uploader("Upload image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="img_uploader")

        if image_file:
            img = Image.open(image_file)
            st.image(img, caption=f"Preview • {image_file.name}", width=900)


            st.caption(f"File: **{image_file.name}** • Size: **{pretty_bytes(len(image_file.getvalue()))}**")

            if st.button("🔍 Analyze Image"):
                if not backend_live:
                    st.error("Backend is offline. Please start FastAPI backend first.")
                else:
                    with st.spinner("Running adversarial analysis..."):
                        img_bytes = image_file.getvalue()
                        files = {"file": (image_file.name, img_bytes, image_file.type or "image/jpeg")}
                        r = requests.post(f"{BACKEND_URL}/detect/image", files=files, timeout=60)

                    if r.status_code == 200:
                        result = r.json()

                        is_adv = bool(result.get("is_adversarial", False))
                        score = result.get("adversarial_score", 0.0)
                        reasons = result.get("reasons", [])

                        if is_adv:
                            st.markdown(f"""
                            <div class="banner-bad">
                              <b>🚨 Adversarial Image Detected</b><br>
                              Suspicion score: <b>{score}</b>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="banner-ok">
                              <b>✅ Clean / Benign Image</b><br>
                              Suspicion score: <b>{score}</b>
                            </div>""", unsafe_allow_html=True)

                        st.write("### 🧾 Reason(s)")
                        for rr in reasons:
                            st.write(f"- {rr}")

                        if show_technical:
                            with st.expander("🔬 Technical Metrics (from backend)"):
                                st.json(result.get("metrics", {}))

                            with st.expander("📦 Full response"):
                                st.json(result)

                    else:
                        st.error(r.text)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧭 How the decision is made")
        st.markdown("""
The backend computes multiple explainable signals:
- **High-frequency energy** (noise-like perturbations)  
- **JPEG blockiness proxy** (artifact patterns)  
- **Entropy** (too noisy / too uniform)  
- **Edge response** (over-sharpening / blur)  
- **Color imbalance** (unusual distribution)
        """)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🎯 Good demo images")
        st.markdown("""
- Normal photos (clean)  
- Highly compressed screenshots (block artifacts)  
- Noisy images / edited images (often flagged)
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# OPERATOR PANEL
# =========================
elif page == "⚙️ Operator Control Panel":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Operator Control Panel")
    st.markdown('<p class="small-muted">Generate dataset + view 20 dynamic samples with adversarial reasons.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])

    with c1:
        if st.button("🧪 Generate Synthetic Dataset"):
            if not backend_live:
                st.error("Backend is offline. Start FastAPI first.")
            else:
                with st.spinner("Generating dataset..."):
                    r = requests.post(f"{BACKEND_URL}/generate_synthetic", timeout=60)
                if r.status_code == 200:
                    st.success("Dataset Generated ✅")
                    st.json(r.json())
                    # auto-refresh samples after generating
                    st.session_state["need_refresh_samples"] = True
                else:
                    st.error(r.text)

    with c2:
        st.markdown("""
        <div class="kpi">
          <div class="label">Dataset Path</div>
          <div class="value">backend/data/synthetic_dataset.csv</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="banner-warn">
          <b>ℹ️ Reminder</b><br>
          Click <b>Refresh Samples</b> to view new combinations each time.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --------- Samples Controls ----------
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        n = st.number_input("Samples to show", min_value=5, max_value=50, value=20, step=5)
    with colB:
        refresh = st.button("🔄 Refresh Samples")

    # auto refresh after generation
    if st.session_state.get("need_refresh_samples"):
        refresh = True
        st.session_state["need_refresh_samples"] = False

    # --------- Fetch sample from backend ----------
    if refresh or ("sample_rows" not in st.session_state):
        if not backend_live:
            st.error("Backend is offline. Please start backend.")
        else:
            with st.spinner("Fetching dynamic samples..."):
                rr = requests.get(f"{BACKEND_URL}/dataset/sample", params={"n": int(n)}, timeout=60)
            if rr.status_code == 200:
                payload = rr.json()
                st.session_state["sample_rows"] = payload.get("rows", [])
            else:
                st.error(rr.text)

    rows = st.session_state.get("sample_rows", [])
    if rows:
        df = pd.DataFrame(rows)

        # Make a cleaner table view
        feature_cols = [c for c in df.columns if c.startswith("f")]
        show_features = feature_cols[:6]  # show first 6 features only for readability

        view_df = df[["source", "label", "is_adversarial", "reason"] + show_features].copy()

        # Add a nice status column
        view_df.insert(0, "status", view_df["is_adversarial"].apply(lambda x: "🚨 Adversarial" if x else "✅ Normal"))

        st.subheader("📌 20 Dynamic Synthetic Examples")
        st.caption("Each refresh shows a new random combination from the dataset.")

        st.dataframe(view_df, height=420)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🔍 Inspect one sample")

        # Select row to inspect
        options = list(range(len(view_df)))
        idx = st.selectbox("Select sample index", options, index=0)

        selected = df.iloc[int(idx)].to_dict()

        if selected.get("is_adversarial"):
            st.markdown(f"""
            <div class="banner-bad">
              <b>🚨 Adversarial Sample</b><br>
              Reason: <b>{selected.get("reason","")}</b><br>
              Source: <b>{selected.get("source","")}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="banner-ok">
              <b>✅ Normal Sample</b><br>
              Reason: <b>{selected.get("reason","")}</b><br>
              Source: <b>{selected.get("source","")}</b>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("See full row (all features)"):
            st.json(selected)

    else:
        st.info("No sample rows loaded yet. Click **Refresh Samples**.")

    st.markdown("</div>", unsafe_allow_html=True)