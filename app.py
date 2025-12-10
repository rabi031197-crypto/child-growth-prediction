import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Child Growth Predictor", layout="wide")

FEATURE_NAMES = ['age','gender_num','height','weight','BMI','B','Mg','P','Se','Si','Zn']

# ---------------------------------------------------------
# SAFE MODEL LOADING (NO CRASH EVEN IF FILES MISSING)
# ---------------------------------------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return None

# Load models directly from main GitHub folder
model_h = safe_load("model_height.joblib")
model_h_2y = safe_load("model_height_2y.joblib")
model_w = safe_load("model_weight.joblib")
model_w_2y = safe_load("model_weight_2y.joblib")
scaler = safe_load("scaler.joblib")

models_loaded = all([
    model_h is not None,
    model_h_2y is not None,
    model_w is not None,
    model_w_2y is not None,
    scaler is not None
])

# ---------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------
st.title("📈 Child Growth Prediction App")
st.markdown("This app predicts **height and weight** 1 and 2 years into the future.")

if not models_loaded:
    st.error("❗ Models could not be loaded. Please upload them below.")
else:
    st.success("✅ Models loaded successfully from repository root.")

# ---------------------------------------------------------
# ALLOW MANUAL MODEL UPLOAD (OPTIONAL)
# ---------------------------------------------------------
with st.expander("📤 Upload model files (optional if GitHub loading fails)"):
    mh = st.file_uploader("model_height.joblib")
    mh2 = st.file_uploader("model_height_2y.joblib")
    mw = st.file_uploader("model_weight.joblib")
    mw2 = st.file_uploader("model_weight_2y.joblib")
    sc = st.file_uploader("scaler.joblib")

    if mh and mh2 and mw and mw2 and sc:
        model_h = joblib.load(mh)
        model_h_2y = joblib.load(mh2)
        model_w = joblib.load(mw)
        model_w_2y = joblib.load(mw2)
        scaler = joblib.load(sc)
        models_loaded = True
        st.success("Models successfully loaded from uploads!")

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.header("Menu")
page = st.sidebar.radio("Choose:", ["Single Prediction", "Batch Prediction"])

# ---------------------------------------------------------
# SINGLE CHILD PREDICTION
# ---------------------------------------------------------
if page == "Single Prediction":
    st.header("👧 Single Child Prediction")

    age = st.number_input("Age (years)", min_value=0, max_value=18, value=5)
    gender = st.selectbox("Gender", ["T", "F"])
    height = st.number_input("Height (cm)", 30.0, 220.0, 110.0)
    weight = st.number_input("Weight (kg)", 1.0, 200.0, 17.0)

    B = st.number_input("Boron (B)", value=1.23)
    Mg = st.number_input("Magnesium (Mg)", value=72.76)
    P = st.number_input("Phosphorus (P)", value=171.17)
    Se = st.number_input("Selenium (Se)", value=0.52)
    Si = st.number_input("Silicon (Si)", value=19.8)
    Zn = st.number_input("Zinc (Zn)", value=179.6)

    if st.button("Predict Growth"):
        if not models_loaded:
            st.error("Models not loaded. Please upload model files.")
        else:
            gender_num = 1 if gender == "T" else 0
            BMI = weight / (height/100)**2

            df = pd.DataFrame([[age, gender_num, height, weight, BMI, B, Mg, P, Se, Si, Zn]],
                              columns=FEATURE_NAMES)

            Xs = scaler.transform(df)

            h1 = float(model_h.predict(Xs)[0])
            h2 = float(model_h_2y.predict(Xs)[0])
            w1 = float(model_w.predict(Xs)[0])
            w2 = float(model_w_2y.predict(Xs)[0])

            st.metric("📏 Height after 1 year", f"{h1:.2f} cm")
            st.metric("📏 Height after 2 years", f"{h2:.2f} cm")
            st.metric("⚖️ Weight after 1 year", f"{w1:.2f} kg")
            st.metric("⚖️ Weight after 2 years", f"{w2:.2f} kg")

            # Plot charts
            years = [0, 1, 2]
            heights = [height, h1, h2]
            weights = [weight, w1, w2]

            fig, ax = plt.subplots()
            ax.plot(years, heights, marker='o')
            ax.set_title("Height Growth Curve")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.plot(years, weights, marker='o')
            ax2.set_title("Weight Growth Curve")
            st.pyplot(fig2)

# ---------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------
if page == "Batch Prediction":
    st.header("📦 Batch Prediction (CSV/XLSX)")

    file = st.file_uploader("Upload file", type=["csv", "xlsx"])

    if file:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

        required = ['age','height','weight','B','Mg','P','Se','Si','Zn']
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        elif not models_loaded:
            st.error("Models not loaded. Upload models first.")
        else:
            if 'gender' in df.columns:
                df['gender_num'] = df['gender'].map({'T':1,'F':0})
            if 'BMI' not in df.columns:
                df['BMI'] = df['weight'] / (df['height']/100)**2

            X = df[FEATURE_NAMES]
            Xs = scaler.transform(X)

            df["height_1y"] = model_h.predict(Xs)
            df["height_2y"] = model_h_2y.predict(Xs)
            df["weight_1y"] = model_w.predict(Xs)
            df["weight_2y"] = model_w_2y.predict(Xs)

            st.dataframe(df.head())

            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            buffer.seek(0)

            st.download_button("Download Predictions.xlsx",
                               buffer,
                               "Predictions.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("App auto-generated — ready for Streamlit deployment.")
