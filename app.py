import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Base URL of your GitHub repository
BASE_URL = "https://raw.githubusercontent.com/rabi031197-crypto/child-growth-prediction/main/"

# -------------------------
# LOAD MODEL PARAMETERS
# -------------------------
def load_json_model(filename):
    """Load GradientBoostingRegressor from JSON params."""
    url = BASE_URL + filename
    response = requests.get(url)

    params = json.loads(response.text)  # JSON → dict

    model = GradientBoostingRegressor()
    model.set_params(**params)
    return model

# -------------------------
# LOAD SCALER PARAMETERS
# -------------------------
def load_scaler(filename):
    url = BASE_URL + filename
    response = requests.get(url)
    params = json.loads(response.text)

    scaler = StandardScaler()
    scaler.set_params(**params)
    return scaler

# -------------------------
# LOAD ALL MODELS + SCALER
# -------------------------
model_h_1y = load_json_model("model_height.json")
model_w_1y = load_json_model("model_weight.json")
model_h_2y = load_json_model("model_height_2y.json")
model_w_2y = load_json_model("model_weight_2y.json")
scaler = load_scaler("scaler.json")

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📈 Child Growth Prediction App")
st.write("Predict height and weight gains after 1–2 years using anthropometric and trace element data.")

age = st.number_input("Age (years)", 1, 18)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Current Height (cm)", 30, 200)
weight = st.number_input("Current Weight (kg)", 2, 150)

# Trace Elements
B = st.number_input("Boron (B)", 0.0, 300.0)
Mg = st.number_input("Magnesium (Mg)", 0.0, 300.0)
P = st.number_input("Phosphorus (P)", 0.0, 300.0)
Se = st.number_input("Selenium (Se)", 0.0, 10.0)
Si = st.number_input("Silicon (Si)", 0.0, 50.0)
Zn = st.number_input("Zinc (Zn)", 0.0, 300.0)

gender_num = 1 if gender == "Male" else 0

# -------------------------
# CREATE INPUT DATAFRAME
# -------------------------
input_df = pd.DataFrame([{
    "age": age,
    "gender": gender_num,
    "height": height,
    "weight": weight,
    "B": B,
    "Mg": Mg,
    "P": P,
    "Se": Se,
    "Si": Si,
    "Zn": Zn
}])

# Scale numerical inputs
X_scaled = scaler.transform(input_df)

# -------------------------
# PREDICT
# -------------------------
pred_h1 = model_h_1y.predict(X_scaled)[0]
pred_w1 = model_w_1y.predict(X_scaled)[0]

pred_h2 = model_h_2y.predict(X_scaled)[0]
pred_w2 = model_w_2y.predict(X_scaled)[0]

# -------------------------
# RESULTS
# -------------------------
st.subheader("📊 Prediction Results")

st.write(f"### 📌 Height gain in 1 year: **{pred_h1 - height:.2f} cm**")
st.write(f"### 📌 Weight gain in 1 year: **{pred_w1 - weight:.2f} kg**")

st.write(f"### 📌 Height gain in 2 years: **{pred_h2 - height:.2f} cm**")
st.write(f"### 📌 Weight gain in 2 years: **{pred_w2 - weight:.2f} kg**")

st.success("Prediction completed successfully!")
