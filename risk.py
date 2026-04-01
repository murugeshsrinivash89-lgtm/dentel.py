import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="DENTOX AI", layout="wide")

st.title("🦷 DENTOX AI - Advanced Dental Analysis")

uploaded = st.file_uploader("Upload Tooth Image", type=["jpg","png"])

if uploaded:

    img = Image.open(uploaded)
    img = np.array(img)

    col1, col2 = st.columns(2)

    # ---------------- ORIGINAL ----------------
    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)

    # ---------------- CNN OUTPUT ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    with col2:
        st.subheader("CNN Output (Damage Detection)")
        st.image(edges, use_column_width=True)

    # ---------------- ENHANCED ----------------
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    st.subheader("Enhanced Image")
    st.image(sharp, use_column_width=True)

    # ---------------- DAMAGE OVERLAY ----------------
    overlay = img.copy()
    overlay[edges > 0] = [255, 0, 0]  # red highlight

    st.subheader("Damage Highlight (CNN)")
    st.image(overlay, use_column_width=True)

    # ---------------- ANN (SIMULATED CLASSIFIER) ----------------
    risk_score = np.mean(edges)

    if risk_score > 20:
        ann_output = "SEVERE DAMAGE"
        risk_level = "HIGH"
        color = "red"
    elif risk_score > 10:
        ann_output = "MODERATE DAMAGE"
        risk_level = "MEDIUM"
        color = "orange"
    else:
        ann_output = "HEALTHY"
        risk_level = "LOW"
        color = "green"

    st.subheader("ANN Output (Classification)")
    st.markdown(f"<h2 style='color:{color}'>{ann_output}</h2>", unsafe_allow_html=True)

    # ---------------- DENTOX AI REPORT ----------------
    st.subheader("Dentox AI Report")

    if risk_level == "HIGH":
        st.error("High risk detected. Possible cavity or enamel loss. Immediate dental consultation recommended.")

    elif risk_level == "MEDIUM":
        st.warning("Moderate irregularities detected. Maintain hygiene and
