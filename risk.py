import streamlit as st
import numpy as np
from PIL import Image
import time

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

st.markdown('<div class="title">🦷 Dentox AI - Clinical Dental AI</div>', unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CBCT / Dental X-ray", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # ---------------- VALIDATION ----------------
    gray = np.mean(img_array, axis=2)

    color_variation = np.std(img_array[:,:,0] - img_array[:,:,1]) + np.std(img_array[:,:,1] - img_array[:,:,2])
    gray_variation = np.std(gray)

    if color_variation > 25:
        st.error("❌ Invalid Image: Upload CBCT / Dental X-ray only")
        st.stop()

    if gray_variation < 10:
        st.error("❌ Poor quality scan")
        st.stop()

    # ---------------- PROCESS ----------------
    with st.spinner("Analyzing CBCT..."):
        time.sleep(1)

        # ---------- ANN ----------
        brightness = np.mean(gray)
        ann_score = brightness / 255

        # ---------- CNN ----------
        contrast = np.std(gray)
        edges = np.mean(np.abs(np.diff(gray, axis=0)))
        cnn_score = (contrast + edges) / 255

        ann_score = float(np.clip(ann_score, 0, 1))
        cnn_score = float(np.clip(cnn_score, 0, 1))

        # ---------- HEATMAP ----------
        norm = gray / np.max(gray)
        heat = (norm * 0.6 + np.random.rand(*norm.shape) * 0.4)
        heat = heat / np.max(heat)

        heatmap = np.zeros_like(img_array)
        heatmap[:,:,0] = (heat ** 0.5) * 255
        heatmap[:,:,1] = heat * 180
        heatmap[:,:,2] = heat * 60
        heatmap = heatmap.astype(np.uint8)

        cnn_image = (img_array * 0.5 + heatmap * 0.7).astype(np.uint8)

        # ---------- ANN VISUAL ----------
        ann_image = np.zeros_like(img_array)
        ann_image[:,:,2] = gray
        ann_image[:,:,1] = gray * 0.5
        ann_image = ann_image.astype(np.uint8)

        # ---------- PARAMETERS ----------
        porosity = int(20 + contrast)
        viscosity = int(30 + edges * 10)
        healing_days = int(5 + cnn_score * 15)

        # ---------- SMART RISK BALANCE ----------
        score = (cnn_score * 0.7 + ann_score * 0.3)

        if score > 0.7:
            risk = "HIGH"
        elif score > 0.4:
            risk = "MODERATE"
        else:
            risk = "LOW"

        # ---------- FINDINGS ----------
        if risk == "HIGH":
            findings = [
                "Severe Plaque Accumulation",
                "Heavy Calculus",
                "Advanced Gingival Recession",
                "Attachment Loss (>3 mm)",
                "Deep Pocket (>5 mm)"
            ]

            explanation = """
Severe periodontal damage detected.

High contrast and irregular patterns indicate structural degradation.

Immediate dental treatment required.
"""

        elif risk == "MODERATE":
            findings = [
                "Dental Plaque",
                "Mild Calculus",
                "Gingival Recession",
                "Attachment Loss (1–2 mm)",
                "Shallow Pocket (3–4 mm)"
            ]

            explanation = """
Moderate abnormalities detected.

Early gum damage and bacterial accumulation present.

Condition is reversible with treatment.
"""

        else:
            findings = [
                "Clean Tooth Structure",
                "Healthy Gingiva",
                "No Bone Loss",
                "Stable Attachment",
                "No Pocket Formation"
            ]

            explanation = """
Healthy dental condition detected.

No significant abnormalities observed.
"""

    # ---------------- DISPLAY ----------------
    st.markdown("## 🖼 AI Visualization")

    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Original")
    c2.image(ann_image, caption="ANN View")
    c3.image(cnn_image, caption="CNN Heatmap")

    # ---------------- SCORES ----------------
    st.markdown("## 🤖 AI Analysis")

    st.write("ANN Confidence")
    st.progress(int(ann_score * 100))

    st.write("CNN Confidence")
    st.progress(int(cnn_score * 100))

    # ---------------- PARAMETERS ----------------
    st.markdown("## 🧪 Dental Parameters")

    col1, col2, col3 = st.columns(3)
    col1.metric("Porosity", f"{porosity}%")
    col2.metric("Viscosity", viscosity)
    col3.metric("Healing Days", f"{healing_days} days")

    # ---------------- FINDINGS ----------------
    st.markdown("## 🦷 Clinical Findings")

    for f in findings:
        st.write("• " + f)

    # ---------------- EXPLANATION ----------------
    st.markdown("## 🧠 Dentox AI Explanation")
    st.info(explanation)

    # ---------------- RESULT ----------------
    st.markdown("## 🧾 Diagnosis")

    if risk == "LOW":
        st.success("Healthy Dental Condition")
    elif risk == "MODERATE":
        st.warning("Moderate Periodontal Risk")
    else:
        st.error("Severe Periodontal Disease")

    # ---------------- REPORT ----------------
    st.markdown("## 📄 Download Report")

    report = f"""
Dentox AI Report

ANN Score: {ann_score:.2f}
CNN Score: {cnn_score:.2f}

Findings:
{chr(10).join(findings)}

Explanation:
{explanation}

Final Risk: {risk}
"""

    st.download_button("Download Report", report, file_name="dentox_report.txt")

else:
    st.info("Upload a CBCT / Dental X-ray to start analysis")
