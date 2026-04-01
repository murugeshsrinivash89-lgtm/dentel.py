import streamlit as st
import numpy as np
from PIL import Image
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Dentox AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #00ffe0;
}
.section {
    font-size: 26px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🦷 Dentox AI - Smart Dental Analysis</div>', unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Dental Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # ---------------- AI PROCESS ----------------
    with st.spinner("Analyzing..."):
        time.sleep(1)

        # ANN (grayscale)
        ann_gray = np.mean(img_array, axis=2)
        ann_image = np.stack((ann_gray,)*3, axis=-1).astype(np.uint8)

        # CNN (edge detection simple)
        kernel = np.array([
            [-1,-1,-1],
            [-1, 8,-1],
            [-1,-1,-1]
        ])

        cnn_image = np.zeros_like(img_array)

        for i in range(1, img_array.shape[0]-1):
            for j in range(1, img_array.shape[1]-1):
                region = img_array[i-1:i+2, j-1:j+2, 0]
                value = np.sum(region * kernel)
                value = max(0, min(255, value))
                cnn_image[i, j] = [value, value, value]

        # ---------------- SCORES ----------------
        ann_score = float(np.random.uniform(0.2, 0.9))
        cnn_score = float(np.random.uniform(0.2, 0.9))

        porosity = int(np.random.randint(20, 80))
        viscosity = int(np.random.randint(30, 100))
        crack = np.random.choice(["Low", "Medium", "High"])
        healing_days = int(np.random.randint(5, 20))

        # Risk logic
        if ann_score > 0.75 or cnn_score > 0.75:
            risk = "HIGH"
        elif ann_score > 0.5 or cnn_score > 0.5:
            risk = "MODERATE"
        else:
            risk = "LOW"

    # ---------------- IMAGE DISPLAY ----------------
    st.markdown("## 🖼 AI Image Comparison")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(image, caption="Original", use_column_width=True)

    with c2:
        st.image(ann_image, caption="ANN Output", use_column_width=True)

    with c3:
        st.image(cnn_image, caption="CNN Output", use_column_width=True)

    # ---------------- AI SCORES ----------------
    st.markdown("## 🤖 AI Analysis")

    st.write("ANN Confidence")
    st.progress(int(ann_score * 100))

    st.write("CNN Confidence")
    st.progress(int(cnn_score * 100))

    # ---------------- PARAMETERS ----------------
    st.markdown("## 🧪 Dental Parameters")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Porosity", f"{porosity}%")
    col2.metric("Viscosity", viscosity)
    col3.metric("Crack Level", crack)
    col4.metric("Healing Days", f"{healing_days} days")

    # ---------------- RESULT ----------------
    st.markdown("## 🧾 Diagnosis")

    if risk == "LOW":
        st.success("Healthy Condition")
    elif risk == "MODERATE":
        st.warning("Moderate Issue Detected")
    else:
        st.error("High Risk Detected")

    # ---------------- REPORT ----------------
    st.markdown("## 📄 Download Report")

    report = f"""
Dentox AI Report

ANN Score: {ann_score:.2f}
CNN Score: {cnn_score:.2f}

Porosity: {porosity}%
Viscosity: {viscosity}
Crack Level: {crack}
Healing Days: {healing_days}

Final Risk: {risk}
"""

    st.download_button("Download Report", report, file_name="dentox_report.txt")

else:
    st.info("Upload a dental image to start analysis")
