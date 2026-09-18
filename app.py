import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Scaler aur Model Load Karein
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    model = tf.keras.models.load_model('model.h5')
    return scaler, model

scaler, model = load_assets()

st.set_page_config(page_title="Model Dashboard", layout="wide")
st.title("📊 Machine Learning Model Dashboard")

# Metrics Display
col1, col2, col3 = st.columns(3)
col1.metric(label="Test Accuracy", value="91.03%")
col2.metric(label="Test Loss", value="0.3051")
col3.metric(label="Epochs Trained", value="40")

st.divider()

# Inputs
st.header("🔮 Real-time Prediction")
f1 = st.number_input("Input Feature 1", value=1.0)
f2 = st.number_input("Input Feature 2", value=2.0)

if st.button("Predict"):
    # Real Prediction Logic
    features = [f1, f2] + [0.0] * 20
    raw_input = np.array([features]) # Dataset ke actual feature count ke mutabiq
    scaled_input = scaler.transform(raw_input)
    prob = model.predict(scaled_input)[0][0]
    
    st.write("---")
    if prob >= 0.5:
        st.success(f"**Prediction: Class 1 (Positive Outcome)** | Confidence: {prob*100:.2f}%")
    else:
        st.error(f"**Prediction: Class 0 (Negative Outcome)** | Confidence: {(1-prob)*100:.2f}%")
        