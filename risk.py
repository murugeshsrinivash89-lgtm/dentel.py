import streamlit as st
import cv2
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore

# ================================
# 🔥 FIREBASE INIT
# ================================
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ================================
# 🧠 DENTOX AI
# ================================
def dentox_ai(risk, porosity, crack):

    reasons = []

    if porosity > 5:
        reasons.append("high porosity")
    elif porosity > 2:
        reasons.append("moderate porosity")
    else:
        reasons.append("low porosity")

    if crack == "YES":
        reasons.append("crack detected")

    txt = ", ".join(reasons)

    if risk < 20:
        return f"Stable tooth. {txt}"

    elif risk < 40:
        return f"Early damage due to {txt}"

    elif risk < 60:
        return f"Moderate decay from {txt}"

    elif risk < 80:
        return f"Severe structural damage due to {txt}"

    else:
        return f"Critical condition due to {txt}. Immediate care needed!"

# ================================
# 💾 SAVE DATA
# ================================
def save_data(name, risk, severity, treatment):

    data = {
        "name": name,
        "risk": risk,
        "severity": severity,
        "treatment": treatment
    }

    db.collection("patients").add(data)

# ================================
# UI
# ================================
st.set_page_config(page_title="Dentox AI", layout="wide")

st.title("🦷 Dentox AI Analysis")

name = st.text_input("Enter Patient Name")

uploaded_file = st.file_uploader("Upload Dental Image")

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 0)

    st.image(image, caption="Original Image", use_container_width=True)

    # ================================
    # PROCESS
    # ================================
    _, teeth_mask = cv2.threshold(image,150,255,cv2.THRESH_BINARY)
    _, raw = cv2.threshold(image,180,255,cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(raw, teeth_mask)

    porosity = (np.sum(mask==255)/image.size)*100

    edges = cv2.Canny(image,50,150)
    crack = "YES" if np.sum(edges)>500 else "NO"

    # risk logic (no dataset version)
    risk = int((porosity*2))

    if crack == "YES":
        risk += 20

    risk = min(risk,100)

    # ================================
    # SEVERITY
    # ================================
    if risk < 20:
        severity="Very Mild"
        treatment="No treatment"
    elif risk < 40:
        severity="Mild"
        treatment="Cleaning"
    elif risk < 60:
        severity="Moderate"
        treatment="Basic care"
    elif risk < 80:
        severity="Severe"
        treatment="Dental procedure"
    else:
        severity="Critical"
        treatment="Immediate dentist"

    explanation = dentox_ai(risk, porosity, crack)

    # ================================
    # DISPLAY RESULT
    # ================================
    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Risk Score", risk)
        st.write("Porosity:", round(porosity,2),"%")
        st.write("Crack:", crack)

    with col2:
        st.write("Severity:", severity)
        st.write("Treatment:", treatment)

    st.info("🧠 Dentox AI: " + explanation)

    # ================================
    # SAVE BUTTON
    # ================================
    if st.button("Save Patient Data"):
        save_data(name, risk, severity, treatment)
        st.success("Saved successfully ✅")
