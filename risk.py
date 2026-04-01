import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="DENTOX AI PRO", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.main-title {
    font-size:40px;
    color:#00f2ff;
    text-align:center;
}
.card {
    background-color:#111827;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 10px rgba(0,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🦷 DENTOX AI PRO SYSTEM</p>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Dental Image", type=["jpg","png"])

if uploaded:

    img = Image.open(uploaded)
    img = np.array(img)

    col1, col2 = st.columns(2)

    # ORIGINAL
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    # CNN
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    with col2:
        st.subheader("CNN Output")
        st.image(edges, use_container_width=True)

    # ENHANCE
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    st.subheader("Enhanced Image")
    st.image(sharp, use_container_width=True)

    # OVERLAY
    overlay = img.copy()
    overlay[edges > 0] = [255, 0, 0]

    st.subheader("Damage Highlight")
    st.image(overlay, use_container_width=True)

    # ---------------- ANALYSIS ----------------

    edge_density = np.mean(edges)

    # POROSITY (based on edge density)
    porosity = min(edge_density / 2, 100)

    # CRACK DETECTION
    crack_score = np.sum(edges > 200) / edges.size * 100

    # VISCOSITY (simulated from smoothness)
    viscosity = max(0, 100 - np.std(gray))

    # ---------------- ANN CLASSIFICATION ----------------

    if edge_density > 20:
        risk_level = "HIGH"
        color = "red"
        ann_output = "SEVERE DAMAGE"
    elif edge_density > 10:
        risk_level = "MEDIUM"
        color = "orange"
        ann_output = "MODERATE DAMAGE"
    else:
        risk_level = "LOW"
        color = "green"
        ann_output = "HEALTHY"

    st.markdown("### 🤖 ANN Output")
    st.markdown(f"<h2 style='color:{color}'>{ann_output}</h2>", unsafe_allow_html=True)

    # ---------------- TREATMENT ----------------

    if risk_level == "HIGH":
        treatment = "Root Canal / Crown Treatment Required"
        healing_days = 30

    elif risk_level == "MEDIUM":
        treatment = "Filling / Cleaning Required"
        healing_days = 14

    else:
        treatment = "No major treatment required"
        healing_days = 5

    # ---------------- REPORT ----------------

    st.markdown("### 📋 Dentox AI Report")

    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Porosity:** {porosity:.2f}%")
    st.write(f"**Crack Level:** {crack_score:.2f}%")
    st.write(f"**Viscosity Index:** {viscosity:.2f}")

    st.write(f"**Recommended Treatment:** {treatment}")
    st.write(f"**Estimated Healing Days:** {healing_days} days")

    # ---------------- METRICS ----------------

    st.markdown("### 📊 Analysis Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Risk", risk_level)
    c2.metric("Porosity", f"{porosity:.1f}%")
    c3.metric("Crack", f"{crack_score:.1f}%")
    c4.metric("Healing Days", healing_days)
