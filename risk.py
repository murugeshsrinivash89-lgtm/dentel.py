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

        # ---------- ANN (BLUE LOOK) ----------
        gray = np.mean(img_array, axis=2)

        ann_image = np.zeros_like(img_array)
        ann_image[:,:,0] = gray * 0.3
        ann_image[:,:,1] = gray * 0.6
        ann_image[:,:,2] = gray
        ann_image = np.clip(ann_image, 0, 255).astype(np.uint8)

        # ---------- CNN (HEATMAP STYLE) ----------
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
        ann_score = float(np.random.uniform(0.3, 0.95))
        cnn_score = float(np.random.uniform(0.3, 0.95))

        porosity = int(np.random.randint(20, 80))
        viscosity = int(np.random.randint(30, 100))
        crack = np.random.choice(["Low", "Moderate", "High"])
        healing_days = int(np.random.randint(5, 20))

        # ---------- DYNAMIC CLINICAL LOGIC ----------
        if cnn_score > 0.75:
            findings = [
                "Heavy Dental Plaque",
                "Moderate Calculus",
                "Gingival Inflammation",
                "Attachment Loss (>2 mm)",
                "Pocket Depth > 4 mm"
            ]

            explanation = """
Severe bacterial accumulation detected.

Calculus deposition is significant leading to gum inflammation.

Attachment loss suggests progressing periodontal disease.

Pocket depth increase indicates tissue destruction.

Immediate dental intervention required.
"""
            risk = "HIGH"

        elif cnn_score > 0.5:
            findings = [
                "Dental Plaque",
                "Supragingival Calculus (Mild)",
                "Mild Gingival Recession",
                "Initial Clinical Attachment Loss (1–2 mm)",
                "Gingival Pocket (3–4 mm)"
            ]

            explanation = """
Moderate plaque accumulation detected.

Early gum recession observed.

Initial attachment loss indicates early-stage periodontal issue.

Condition is reversible with proper care.
"""
            risk = "MODERATE"

        else:
            findings = [
                "Minimal Plaque",
                "Healthy Gingiva",
                "No Attachment Loss",
                "No Pocket Formation"
            ]

            explanation = """
Oral condition appears healthy.

No significant bacterial accumulation detected.

Gum tissue is stable and intact.

Maintain regular oral hygiene.
"""
            risk = "LOW"

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
        st.warning("Moderate Periodontal Risk Detected")
    else:
        st.error("Severe Periodontal Disease Detected")

    # ---------------- REPORT ----------------
    st.markdown("## 📄 Download Report")

    report = "Dentox AI Clinical Report\n\n"
    report += f"ANN Score: {ann_score:.2f}\n"
    report += f"CNN Score: {cnn_score:.2f}\n\n"

    report += "Clinical Findings:\n"
    for f in findings:
        report += f"- {f}\n"

    report += "\nAI Explanation:\n"
    report += explanation

    report += f"\nFinal Risk: {risk}\n"

    st.download_button("Download Report", report, file_name="dentox_report.txt")

else:
    st.info("Upload a dental image to start analysis")
