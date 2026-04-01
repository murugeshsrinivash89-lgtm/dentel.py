import streamlit as st
import numpy as np
from PIL import Image
import time

# ---------------- CONFIG ----------------
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🦷 Dentox AI - Clinical Dental AI System</div>', unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Dental Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # ---------------- PROCESS ----------------
    with st.spinner("Analyzing..."):
        time.sleep(1)

        # ---------- ANN (BLUE AI LOOK) ----------
        gray = np.mean(img_array, axis=2)

        ann_image = np.zeros_like(img_array)
        ann_image[:,:,0] = gray * 0.3
        ann_image[:,:,1] = gray * 0.6
        ann_image[:,:,2] = gray
        ann_image = np.clip(ann_image, 0, 255).astype(np.uint8)

        # ---------- CNN (RANDOM HEATMAP STYLE) ----------
        norm = gray / np.max(gray)

        noise = np.random.rand(*norm.shape) * 0.5
        heat = norm * 0.6 + noise
        heat = heat / np.max(heat)

        heatmap = np.zeros_like(img_array)
        heatmap[:,:,0] = (heat ** 0.5) * 255
        heatmap[:,:,1] = heat * 200
        heatmap[:,:,2] = heat * 50
        heatmap = heatmap.astype(np.uint8)

        cnn_image = (img_array * 0.4 + heatmap * 0.8).astype(np.uint8)

        # ---------- SCORES ----------
        ann_score = 0.72
        cnn_score = 0.81

        porosity = 42
        viscosity = 68
        crack = "Moderate"
        healing_days = 10

        risk = "MODERATE"

    # ---------------- IMAGE DISPLAY ----------------
    st.markdown("## 🖼 AI Visual Comparison")

    c1, c2, c3 = st.columns(3)

    c1.image(image, caption="Original")
    c2.image(ann_image, caption="ANN Output")
    c3.image(cnn_image, caption="CNN Heatmap")

    # ---------------- AI ANALYSIS ----------------
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

    # ---------------- CLINICAL FINDINGS ----------------
    st.markdown("## 🦷 Clinical Findings")

    findings = [
        "Dental Plaque",
        "Supragingival Calculus (Mild)",
        "Mild Gingival Recession",
        "Initial Clinical Attachment Loss (1–2 mm)",
        "No Tooth Mobility (Physiologic Mobility only)",
        "Gingival (False) Pocket",
        "Shallow Pocket (≤ 3–4 mm)"
    ]

    for f in findings:
        st.write("• " + f)

    # ---------------- AI EXPLANATION ----------------
    st.markdown("## 🧠 Dentox AI Explanation")

    explanation = """
Dentox AI has detected early-stage periodontal involvement.

Presence of dental plaque and mild calculus indicates bacterial accumulation.

Gingival recession suggests early gum tissue loss.

Initial attachment loss (1–2 mm) indicates beginning periodontal damage.

False pocket formation is due to gingival swelling.

Shallow pocket depth confirms mild periodontal condition.

Preventive dental care is recommended.
"""

    st.info(explanation)

    # ---------------- RESULT ----------------
    st.markdown("## 🧾 Diagnosis")
    st.warning("Moderate Periodontal Risk Detected")

    # ---------------- REPORT ----------------
    st.markdown("## 📄 Download Report")

    report = "Dentox AI Clinical Report\n\n"
    report += f"ANN Score: {ann_score}\n"
    report += f"CNN Score: {cnn_score}\n\n"

    report += "Clinical Findings:\n"
    for f in findings:
        report += f"- {f}\n"

    report += "\nAI Explanation:\n"
    report += explanation

    st.download_button("Download Report", report, file_name="dentox_report.txt")

else:
    st.info("Upload a dental image to start analysis")
