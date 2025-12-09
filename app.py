import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from xgboost import XGBRegressor

RAW_URL = "https://raw.githubusercontent.com/rabi031197-crypto/child-growth-prediction/main/"

def load_json_xgb(filename):
    url = RAW_URL + filename
    response = requests.get(url)
    params = json.loads(response.text)
    model = XGBRegressor()
    model.set_params(**params)
    return model

# Load models
model_h_1y = load_json_xgb("model_height.json")
model_w_1y = load_json_xgb("model_weight.json")
model_h_2y = load_json_xgb("model_height_2y.json")
model_w_2y = load_json_xgb("model_weight_2y.json")

st.title("Child Growth Prediction App")

age = st.number_input("Age", 1, 18)
gender = st.selectbox("Gender", ["M", "F"])
height = st.number_input("Current Height (cm)", 30, 200)
weight = st.number_input("Current Weight (kg)", 2, 150)

gender_num = 1 if gender == "M" else 0

input_data = pd.DataFrame([{
    "age": age,
    "gender": gender_num,
    "height": height,
    "weight": weight
}])

# Predictions
pred_h1 = model_h_1y.predict(input_data)[0]
pred_w1 = model_w_1y.predict(input_data)[0]
pred_h2 = model_h_2y.predict(input_data)[0]
pred_w2 = model_w_2y.predict(input_data)[0]

st.subheader("Growth Prediction")

st.write(f"📌 **Height gain in 1 year:** {pred_h1 - height:.2f} cm")
st.write(f"📌 **Weight gain in 1 year:** {pred_w1 - weight:.2f} kg")
st.write(f"📌 **Height gain in 2 years:** {pred_h2 - height:.2f} cm")
st.write(f"📌 **Weight gain in 2 years:** {pred_w2 - weight:.2f} kg")
