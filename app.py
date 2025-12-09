import streamlit as st
import pandas as pd
import requests
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

RAW_URL = "https://raw.githubusercontent.com/rabi031197-crypto/child-growth-prediction/main/"

# ------------------ Loading models ------------------

def load_json_model(filename):
    """Load Scikit-Learn model parameters from JSON"""
    url = RAW_URL + filename
    response = requests.get(url)
    params = json.loads(response.text)
    model = GradientBoostingRegressor()
    model.set_params(**params)
    return model

def load_scaler(filename):
    """Load scaler parameters from JSON"""
    url = RAW_URL + filename
    response = requests.get(url)
    params = json.loads(response.text)
    scaler = StandardScaler()
    scaler.set_params(**params)
    return scaler

# Load all models
model_h_1y = load_json_model("model_height.json")
model_h_2y = load_json_model("model_height_2y.json")
model_w_1y = load_json_model("model_weight.json")
model_w_2y = load_json_model("model_weight_2y.json")

# Load scaler
scaler = load_scaler("scaler.json")

# ------------------ UI ------------------

st.title("📈 Child Growth Prediction App")

age = st.number_input("Child Age (years)", min_value=1, max_value=18)
gender = st.selectbox("Gender", ["M", "F"])
height = st.number_input("Current Height (cm)", min_value=30, max_value=200)
weight = st.number_input("Current Weight (kg)", min_value=2, max_value=150)

gender_num = 1 if gender == "M" else 0

# Prepare data
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender_num,
    "height": height,
    "weight": weight
}])

X_scaled = scaler.transform(input_data)

# ------------------ Predictions ------------------

pred_h1 = model_h_1y.predict(X_scaled)[0]
pred_h2 = model_h_2y.predict(X_scaled)[0]

pred_w1 = model_w_1y.predict(X_scaled)[0]
pred_w2 = model_w_2y.predict(X_scaled)[0]

# ------------------ Output ------------------

st.subheader("🌟 Growth Prediction Results")

st.write(f"📌 **Height gain in 1 year:** {pred_h1 - height:.2f} cm")
st.write(f"📌 **Weight gain in 1 year:** {pred_w1 - weight:.2f} kg")

st.write(f"🌳 **Height gain in 2 years:** {pred_h2 - height:.2f} cm")
st.write(f"🌳 **Weight gain in 2 years:** {pred_w2 - weight:.2f} kg")



st.write(f"📌 **Height gain in 1 year:** {pred_h1 - height:.2f} cm")
st.write(f"📌 **Weight gain in 1 year:** {pred_w1 - weight:.2f} kg")
st.write(f"📌 **Height gain in 2 years:** {pred_h2 - height:.2f} cm")
st.write(f"📌 **Weight gain in 2 years:** {pred_w2 - weight:.2f} kg")
