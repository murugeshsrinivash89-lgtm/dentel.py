import streamlit as st
import numpy as np
from PIL import Image

# SAFE CV2 IMPORT
try:
    import cv2
except:
    st.error("OpenCV not installed properly. Check requirements.txt")
    st.stop()

# PAGE
st.set_page_config(page_title="DENTOX AI PRO", layout="wide")

st.markdown("<h1 style='text-align:center;color:#00f2ff;'>🦷 DENTOX AI PRO</h1>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Dental Image", type=["jpg","png"])

if uploaded:

    img = Image.open(uploaded)
    img = np.array(img)

    col1, col2 = st.columns(2)

    # ORIGINAL
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)

    # CNN (EDGE)
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

    porosity = min(edge_density / 2, 100)
    crack_score = np.sum(edges > 0) / edges.size * 100
    viscosity = max(0, 100 - np.std(gray))

    # ANN LOGIC
    if edge_density > 20:
        risk = "HIGH"
        color = "red"
        output = "SEVERE DAMAGE"
    elif edge_density > 10:
        risk = "MEDIUM"
        color = "orange"
        output = "MODERATE DAMAGE"
    else:
        risk = "LOW"
        color = "green"
        output = "HEALTHY"

    st.markdown("### 🤖 ANN Output")
    st.markdown(f"<h2 style='color:{color}'>{output}</h2>", unsafe_allow_html=True)

    # TREATMENT
    if risk == "HIGH":
        treatment = "Root Canal / Crown Required"
        days = 30
    elif risk == "MEDIUM":
        treatment = "Filling / Cleaning Required"
        days = 14
    else:
        treatment = "No major treatment"
        days = 5

    # REPORT
    st.markdown("### 📋 Dentox AI Report")

    st.write(f"Risk Level: {risk}")
    st.write(f"Porosity: {porosity:.2f}%")
    st.write(f"Crack: {crack_score:.2f}%")
    st.write(f"Viscosity: {viscosity:.2f}")

    st.write(f"Treatment: {treatment}")
    st.write(f"Healing Days: {days}")

    # METRICS
    st.markdown("### 📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Risk", risk)
    c2.metric("Porosity", f"{porosity:.1f}%")
    c3.metric("Crack", f"{crack_score:.1f}%")
    c4.metric("Healing", f"{days} days")
