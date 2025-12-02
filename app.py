import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

st.title("Child Growth Prediction App")
st.write("Predict height and weight gain after 1 year and 2 years.")

# ------------------------------
# Helper: load files from GitHub
# ------------------------------
def load_from_github(url):
    response = requests.get(url)
    return joblib.load(BytesIO(response.content))

# ------------------------------
# Load models and scaler
# ------------------------------
# 🔥 IMPORTANT: CHANGE THESE LINKS TO YOUR GITHUB LINKS

BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/"

model_h_1y = load_from_github(BASE_URL + "model_height.joblib")
model_w_1y = load_from_github(BASE_URL + "model_weight.joblib")
model_h_2y = load_from_github(BASE_URL + "model_height_2y.joblib")
model_w_2y = load_from_github(BASE_URL + "model_weight_2y.joblib")
scaler      = load_from_github(BASE_URL + "scaler.joblib")

st.success("Models loaded successfully!")

# ------------------------------
# Input fields
# ------------------------------
st.header("Enter child data")

age = st.number_input("Age (years)", min_value=1, max_value=18, value=7)
gender = st.radio("Gender", ["Male", "Female"])
gender_num = 1 if gender == "Male" else 0

height = st.number_input("Current Height (cm)", min_value=40.0, max_value=200.0, value=120.0)
weight = st.number_input("Current Weight (kg)", min_value=5.0, max_value=120.0, value=25.0)

st.subheader("Trace Elements (Hair)")

B  = st.number_input("Boron (B)",  min_value=0.0, value=1.5)
Mg = st.number_input("Magnesium (Mg)", min_value=0.0, value=30.0)
P  = st.number_input("Phosphorus (P)", min_value=0.0, value=150.0)
Se = st.number_input("Selenium (Se)", min_value=0.0, value=0.45)
Si = st.number_input("Silicon (Si)", min_value=0.0, value=15.0)
Zn = st.number_input("Zinc (Zn)", min_value=0.0, value=90.0)

# ------------------------------
# Create feature vector
# ------------------------------
BMI = weight / (height/100)**2

child = pd.DataFrame([{
    "age": age,
    "gender_num": gender_num,
    "height": height,
    "weight": weight,
    "BMI": BMI,
    "B": B,
    "Mg": Mg,
    "P": P,
    "Se": Se,
    "Si": Si,
    "Zn": Zn
}])

features = ["age","gender_num","height","weight","BMI","B","Mg","P","Se","Si","Zn"]

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Growth"):

    X_scaled = scaler.transform(child[features])

    # Predictions
    pred_h_1y = model_h_1y.predict(X_scaled)[0]
    pred_w_1y = model_w_1y.predict(X_scaled)[0]
    pred_h_2y = model_h_2y.predict(X_scaled)[0]
    pred_w_2y = model_w_2y.predict(X_scaled)[0]

    # Gains
    gain_h_1y = pred_h_1y - height
    gain_w_1y = pred_w_1y - weight
    gain_h_2y = pred_h_2y - height
    gain_w_2y = pred_w_2y - weight

    # ------------------------------
    # Output
    # ------------------------------
    st.header("Predicted Growth")

    st.subheader("📅 Growth in 1 Year")
    st.write(f"**Height gain:** {gain_h_1y:.2f} cm")
    st.write(f"**Weight gain:** {gain_w_1y:.2f} kg")

    st.subheader("📅 Growth in 2 Years")
    st.write(f"**Height gain:** {gain_h_2y:.2f} cm")
    st.write(f"**Weight gain:** {gain_w_2y:.2f} kg")

    st.success("Prediction complete!")

