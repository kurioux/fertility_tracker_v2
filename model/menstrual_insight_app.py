import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- Load Model Artifacts ---
scaler = joblib.load("scaler.pkl")
model = joblib.load("kmeans_model.pkl")  # Consistent naming
mapper = joblib.load("fertility_cluster_mapper.pkl")

# --- Prediction Function ---
def predict_phase(input_data):
    required_features = ['BodyTemp', 'BPM', 'SPO2', 'BodyTemp_RollMean', 'BPM_RollMean', 'Temp_Amplitude']
    input_df = pd.DataFrame([input_data])[required_features]
    input_scaled = scaler.transform(input_df)
    cluster = model.predict(input_scaled)[0]
    phase = mapper.get(cluster, "Unknown")
    return cluster, phase

# --- Streamlit UI Config ---
st.set_page_config(page_title="Experimental Menstrual Advisor", page_icon="🧬", layout="centered")
st.title("🔍 Menstrual Cycle Insight Advisor (Experimental)")

# --- About Section ---
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This experimental module predicts your current menstrual phase and attempts to estimate the exact day in your cycle based on your sensor data and personal inputs. 
    It provides tailored psychological and wellbeing advice to help you make sense of your cycle.
    """)

# --- Input Section ---
st.subheader("📥 Sensor Data & Date")
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡️ Body Temperature (°C)", min_value=35.0, max_value=40.0, value=36.6)
    bpm = st.number_input("💓 Heart Rate (BPM)", min_value=50, max_value=150, value=75)
with col2:
    spo2 = st.number_input("🫁 SpO2 (%)", min_value=90, max_value=100, value=98)
    start_date = st.date_input("📅 First day of your last period")

# --- Lifestyle Factors ---
st.subheader("⚙️ Lifestyle & Physical Factors")
st.markdown("Select the options that match your current state:")
diet = st.radio("🍽️ Diet Quality", ["Healthy & Regular", "Irregular Meals", "High Sugar/Processed"])
sleep = st.radio("😴 Sleep Pattern", ["Well Rested", "Interrupted", "Poor/Short"])
stress = st.radio("😣 Current Stress Level", ["Low", "Moderate", "High"])

# --- Predict Button ---
if st.button("🔎 Analyze & Predict"):
    today = datetime.date.today()
    days_elapsed = (today - start_date).days % 28

    # Buffer calculations (simple placeholders for now)
    temp_buffer = [temperature] * 5
    bpm_buffer = [bpm] * 5
    temp_roll = np.mean(temp_buffer)
    bpm_roll = np.mean(bpm_buffer)
    temp_amp = np.max(temp_buffer) - np.min(temp_buffer)

    input_vector = {
        'BodyTemp': temperature,
        'BPM': bpm,
        'SPO2': spo2,
        'BodyTemp_RollMean': temp_roll,
        'BPM_RollMean': bpm_roll,
        'Temp_Amplitude': temp_amp
    }

    cluster, phase = predict_phase(input_vector)

    # Lifestyle-based modifier
    modifier = 0
    if diet == "High Sugar/Processed":
        modifier += 1
    if sleep == "Poor/Short":
        modifier += 1
    if stress == "High":
        modifier += 1
    estimated_day = (days_elapsed + modifier) % 28

    # --- Output ---
    st.success(f"**Predicted Phase:** {phase}\n**Estimated Cycle Day:** Day {estimated_day + 1}")

    st.markdown("---")
    st.subheader("🧠 Interpretation")
    st.markdown(f"""
    - Based on your input, today is likely **Day {estimated_day + 1}** of your menstrual cycle.
    - Adjustments were made for stress, sleep, and diet, which can shift your hormonal patterns.
    - This estimation helps understand your energy, emotions, and fertility patterns better.
    """)

    # --- Psychological Insights ---
    st.subheader("💡 Wellbeing & Emotional Guidance")
    if estimated_day < 5:
        st.info("Menstrual phase: Focus on rest, comfort foods, and self-compassion.")
    elif estimated_day < 14:
        st.success("Follicular phase: Great time for new ideas, physical activity, and planning ahead.")
    elif estimated_day < 17:
        st.warning("Ovulation: You may feel energetic but also emotionally sensitive—stay grounded.")
    else:
        st.info("Luteal phase: Slow down. Self-care and reflection are helpful now.")

    # --- Symptom Tracker ---
    st.markdown("---")
    st.subheader("📋 Symptom Tracker")
    symptoms = st.multiselect(
        "What symptoms are you experiencing today?",
        ["Cramps", "Bloating", "Headache", "Mood Swings", "Fatigue", "Breast Tenderness", "Backache"]
    )

    if symptoms:
        st.markdown("### 🌿 Suggested Relief Tips")
        for symptom in symptoms:
            if symptom == "Cramps":
                st.write("🧘 Try gentle yoga or apply a heating pad.")
            elif symptom == "Headache":
                st.write("💧 Drink more water and try resting in a dark room.")
            elif symptom == "Mood Swings":
                st.write("📝 Journaling or talking to a friend can help stabilize emotions.")
            elif symptom == "Fatigue":
                st.write("😴 Prioritize sleep and reduce screen time.")
            elif symptom == "Bloating":
                st.write("🥗 Eat smaller meals and reduce sodium.")
            elif symptom == "Breast Tenderness":
                st.write("🧴 Use a warm compress and wear comfortable bras.")
            elif symptom == "Backache":
                st.write("🛀 A warm bath or gentle stretching can help ease back pain.")

    st.caption("🔬 This experimental tool does not replace medical guidance. Consult a healthcare professional for concerns.")
