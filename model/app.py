import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime

# --- Load Model Artifacts ---
scaler = joblib.load("scaler.pkl")
model = joblib.load("kmeans_model.pkl")
mapper = joblib.load("fertility_cluster_mapper.pkl")

def predict_phase(input_data):
    required_features = ['BodyTemp', 'BPM', 'SPO2', 'BodyTemp_RollMean', 'BPM_RollMean', 'Temp_Amplitude']
    input_df = pd.DataFrame([input_data])[required_features]
    input_scaled = scaler.transform(input_df)
    cluster = model.predict(input_scaled)[0]
    phase = mapper.get(cluster, "Unknown")
    return cluster, phase

# --- Streamlit UI Config ---
st.set_page_config(page_title="Menstrual Phase Predictor", page_icon="🌙", layout="centered")
st.title("🌸 Menstrual Phase Predictor")
st.markdown("Enter your health metrics to predict your menstrual phase and learn more about it.")

with st.expander("ℹ️ About This App"):
    st.markdown("""
    This app uses sensor readings to predict your current menstrual cycle phase using machine learning.
    ### Phases Overview:
    - **Follicular Phase**: Begins on the first day of your period. Hormones start rising.  
      🟣 *Basal temperature* is usually lower (~36.4°C).  
      ❤️ *BPM* and *SPO2* tend to be stable.

    - **Ovulation**: A sharp rise in LH hormone occurs.  
      🔺 Basal body temperature rises slightly (~36.6–37.0°C).

    - **Luteal Phase**: Hormones stay high preparing for pregnancy.  
      🌡️ Body temperature stays higher (~36.8°C+).  
      ❤️ BPM may increase slightly.

    - **Menstruation**: Hormone levels drop.  
      ❄️ Temperature and BPM gradually decrease.

    This app can give insight—but isn't a replacement for medical advice.
    """)

# --- Initialize Buffers in Session State ---
if "temp_buffer" not in st.session_state:
    st.session_state.temp_buffer = []
if "bpm_buffer" not in st.session_state:
    st.session_state.bpm_buffer = []

# --- Inputs ---
st.subheader("📥 Sensor Inputs")

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input("🌡️ Body Temperature (°C)", 35.0, 40.0, 36.8, 0.1)
with col2:
    bpm = st.number_input("💓 Heart Rate (BPM)", 50, 150, 78)
with col3:
    spo2 = st.number_input("🫁 SPO2 (%)", 90, 100, 97)

# --- Predict Button ---
if st.button("🔍 Predict Phase"):

    # Update Buffers (max 5 values)
    st.session_state.temp_buffer.append(temp)
    st.session_state.bpm_buffer.append(bpm)
    if len(st.session_state.temp_buffer) > 5:
        st.session_state.temp_buffer.pop(0)
    if len(st.session_state.bpm_buffer) > 5:
        st.session_state.bpm_buffer.pop(0)

    # Rolling Features
    bodytemp_roll = np.mean(st.session_state.temp_buffer)
    bpm_roll = np.mean(st.session_state.bpm_buffer)
    temp_amp = np.max(st.session_state.temp_buffer) - np.min(st.session_state.temp_buffer)

    input_vector = {
        'BodyTemp': temp,
        'BPM': bpm,
        'SPO2': spo2,
        'BodyTemp_RollMean': bodytemp_roll,
        'BPM_RollMean': bpm_roll,
        'Temp_Amplitude': temp_amp
    }

    # Predict
    cluster, phase = predict_phase(input_vector)
    st.success(f"**Predicted Phase:** {phase}  \n(Cluster ID: {cluster})")

    # Save prediction
    log_entry = pd.DataFrame([[datetime.datetime.now(), temp, bpm, spo2, phase]],
                             columns=["Timestamp", "BodyTemp", "BPM", "SPO2", "Phase"])
    log_entry.to_csv("prediction_log.csv", mode="a", header=False, index=False)
    st.info("Prediction saved to `prediction_log.csv` ✅")

# --- Optional Buffer Display ---
with st.expander("🧪 View Rolling Buffer (Last 5 Readings)"):
    st.write("**Temperature Buffer:**", st.session_state.temp_buffer)
    st.write("**Heart Rate Buffer:**", st.session_state.bpm_buffer)
