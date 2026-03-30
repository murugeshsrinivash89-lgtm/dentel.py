import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import random
import time

st.set_page_config(page_title="Dental AI", layout="wide")

# ---------- TITLE ----------
st.markdown("<h1 style='text-align:center;color:#00e6e6;'>DENTAL AI ANALYSIS</h1>", unsafe_allow_html=True)

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("Upload Dental Scan", type=["jpg","png","jpeg"])

# ---------- FUNCTIONS ----------
def generate_fake_metrics():
    return {
        "damage": round(random.uniform(5, 90),2),
        "porosity": round(random.uniform(1, 10),2),
        "density": round(random.uniform(1400, 4000),2),
        "crack": random.choice(["YES","NO"])
    }

def risk_score(damage, porosity, crack):
    score = damage*0.5 + porosity*5 + (20 if crack=="YES" else 0)
    return int(min(score,100))

def severity(score):
    if score < 30: return "Mild"
    elif score < 60: return "Moderate"
    elif score < 80: return "Severe"
    else: return "Very Severe"

def treatment(level):
    if level=="Mild": return "Cleaning & monitoring"
    elif level=="Moderate": return "Basic dental care"
    elif level=="Severe": return "Dentist checkup required"
    else: return "Immediate treatment required"

def dentox_ai(damage, porosity, crack):
    if damage < 30:
        return "Minor damage. Maintain hygiene."
    elif damage < 60:
        return "Moderate wear detected. Early treatment advised."
    else:
        return "Severe damage. Risk of cavity or fracture."

# ---------- MAIN ----------
if uploaded_file:

    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(img, caption="Uploaded Scan", use_container_width=True)

        # fake heatmap animation
        placeholder = st.empty()
        for i in range(5):
            time.sleep(0.3)
            noise = np.random.rand(300,300)
            placeholder.image(noise, caption="Scanning...", use_container_width=True)

    # ---------- DATA ----------
    data = []
    dentox_messages = []

    for i in range(10):
        m = generate_fake_metrics()
        r = risk_score(m["damage"], m["porosity"], m["crack"])
        s = severity(r)
        t = treatment(s)

        data.append({
            "Tooth": f"T{i+1}",
            "Damage %": m["damage"],
            "Porosity": m["porosity"],
            "Density": m["density"],
            "Crack": m["crack"],
            "Risk": r,
            "Severity": s,
            "Treatment": t
        })

        dentox_messages.append(f"T{i+1}: {dentox_ai(m['damage'], m['porosity'], m['crack'])}")

    df = pd.DataFrame(data)

    # ---------- UI ----------
    with col2:
        st.dataframe(df, use_container_width=True)

    # ---------- DENTOX AI ----------
    st.markdown("## 🤖 Dentox AI Insights")
    for msg in dentox_messages:
        st.write("•", msg)

else:
    st.info("Upload a dental scan to start analysis")
