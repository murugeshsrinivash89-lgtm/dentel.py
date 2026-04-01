import streamlit as st
import numpy as np
from PIL import Image
import time

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Dentox AI 🦷", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #00ffe0;
    text-align: center;
}
.card {
    background-color: #111;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px #00ffe0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🦷 Dentox AI - Dental Analysis System</div>', unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Dental Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- SIMULATED ANALYSIS ----------------
    with st.spinner("Analyzing..."):
        time.sleep(2)

        # Fake AI outputs
        ann_score = np.random.uniform(0, 1)
        cnn_score = np.random.uniform(0, 1)

        porosity = np.random.randint(10, 90)
        viscosity = np.random.randint(20, 100)
        crack = np.random.choice(["Low", "Medium", "High"])
        healing_days = np.random.randint(3, 20)

        # Risk logic
        risk_level = "LOW"
        if ann_score > 0.6 or cnn_score > 0.6:
            risk_level = "MODERATE"
        if ann_score > 0.8 or cnn_score > 0.8:
            risk_level = "HIGH"

    with col2:
        st.markdown("### 🤖 AI Analysis")

        st.progress(int(ann_score * 100))
        st.write(f"ANN Confidence: {ann_score:.2f}")

        st.progress(int(cnn_score * 100))
        st.write(f"CNN Confidence: {cnn_score:.2f}")

    # ---------------- PARAMETERS ----------------
    st.markdown("## 🧪 Dental Parameters")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Porosity", f"{porosity}%")
    c2.metric("Viscosity", f"{viscosity}")
    c3.metric("Crack Level", crack)
    c4.metric("Healing Days", f"{healing_days} days")

    # ---------------- RESULT ----------------
    st.markdown("## 🧾 Final Diagnosis")

    if risk_level == "LOW":
        st.success("✅ Healthy Condition")
    elif risk_level == "MODERATE":
        st.warning("⚠ Moderate Issue Detected")
    else:
        st.error("🚨 High Risk Detected")

    # ---------------- REPORT ----------------
    st.markdown("## 📄 Dentox AI Report")

    report = f"""
    Dental Analysis Report

    ANN Score: {ann_score:.2f}
    CNN Score: {cnn_score:.2f}

    Porosity: {porosity}%
    Viscosity: {viscosity}
    Crack Level: {crack}
    Healing Days: {healing_days}

    Final Risk: {risk_level}
    """

    st.download_button("Download Report", report, file_name="dentox_report.txt")

else:
    st.info("Upload a dental image to start analysis")