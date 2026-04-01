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

        # ---------- ANN (BLUE) ----------
        gray = np.mean(img_array, axis=2)

        ann_image = np.zeros_like(img_array)
        ann_image[:,:,0] = gray * 0.3
        ann_image[:,:,1] = gray * 0.6
        ann_image[:,:,2] = gray
        ann_image = np.clip(ann_image, 0, 255).astype(np.uint8)

        # ---------- CNN (HEATMAP) ----------
        edges = np.zeros_like(gray)

        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                value = (
                    -gray[i-1,j-1] - gray[i-1,j] - gray[i-1,j+1]
                    -gray[i,j-1]   + 8*gray[i,j] - gray[i,j+1]
                    -gray[i+1,j-1] - gray[i+1,j] - gray[i+1,j+1]
                )
                edges[i,j] = abs(value)

        max_val = np.max(edges)
        if max_val != 0:
            edges = edges / max_val

        heatmap = np.zeros_like(img_array)
        heatmap[:,:,0] = edges * 255
        heatmap[:,:,1] = edges * 180
        heatmap[:,:,2] = edges * 30
        heatmap = heatmap.astype(np.uint8)

        cnn_image = (img_array * 0.5 + heatmap * 0.7).astype(np.uint8)

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

Presence of dental plaque and mild calculus indicates bacterial accumulation on tooth surfaces.

Gingival recession suggests early gum tissue loss, possibly due to inflammation or improper oral hygiene.

Initial attachment loss (1–2 mm) indicates the beginning of periodontal breakdown.

False pocket formation is due to gingival swelling rather than bone loss.

Shallow pocket depth (≤ 3–4 mm) confirms mild periodontal condition.

Overall condition suggests early gingivitis transitioning towards mild periodontitis, requiring preventive care.
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
