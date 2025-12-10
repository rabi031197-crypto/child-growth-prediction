# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Child Growth Predictor / Баланың өсуі", layout="wide")

FEATURE_NAMES = ['age','gender_num','height','weight','BMI','B','Mg','P','Se','Si','Zn']

# --- Translations (Kazakh simple, Russian, English) ---
TXT = {
    "lang_names": ["Қазақша", "Русский", "English"],
    "title": [
        "Балалардың өсуін болжау (1 және 2 жыл)",        # Kazakh
        "Прогноз роста детей (1 и 2 года)",            # Russian
        "Child Growth Prediction (1 & 2 years)"       # English
    ],
    "subtitle": [
        "Бала туралы деректерді енгізіп, 1 және 2 жылдағы бой мен салмақты көріңіз.",
        "Введите данные ребенка и получите прогноз роста и веса через 1 и 2 года.",
        "Enter child data and get height & weight predictions in 1 and 2 years."
    ],
    "models_loaded": [
        "✅ Модельдер жүктелді (репозиторий түбірінен).",
        "✅ Модели загружены (из корня репозитория).",
        "✅ Models loaded (from repository root)."
    ],
    "models_not_loaded": [
        "❗ Модельдер жүктелмеді. Оларды жүктеңіз немесе \"Файлдардан жүктеу\" бөлімінен жүктеңіз.",
        "❗ Модели не загружены. Загрузите их или используйте раздел \"Загрузить файлы\".",
        "❗ Models not loaded. Upload them or use the \"Upload files\" section."
    ],
    "upload_expander": [
        "📤 Файлдардан жүктеу (егер GitHub-тен жүктелмесе)",
        "📤 Загрузить файлы (если не загружено из GitHub)",
        "📤 Upload files (if not loaded from GitHub)"
    ],
    "upload_hint": [
        "Егер модельдер репозиторийде болмаса немесе Git LFS мәселесі болса, мұнда joblib файлдарын жүктеңіз.",
        "Если модели не в репозитории или есть проблемы с LFS, загрузите joblib файлы здесь.",
        "If models are missing in the repo or you face LFS issues, upload joblib files here."
    ],
    "single_pred": ["Жеке болжам", "Одиночный прогноз", "Single prediction"],
    "batch_pred": ["Көші-қон (CSV/XLSX)", "Пакетный прогноз (CSV/XLSX)", "Batch prediction (CSV/XLSX)"],
    "age": ["Жасы (жыл)", "Возраст (лет)", "Age (years)"],
    "gender": ["Жынысы", "Пол", "Gender"],
    "gender_options": [["Ер", "Қыз"], ["Мальчик", "Девочка"], ["Boy", "Girl"]],
    "height": ["Бойы (см)", "Рост (см)", "Height (cm)"],
    "weight": ["Салмағы (кг)", "Вес (кг)", "Weight (kg)"],
    "B": ["B (бор)", "B (бор)", "B (boron)"],
    "Mg": ["Mg (магний)", "Mg (магний)", "Mg (magnesium)"],
    "P": ["P (фосфор)", "P (фосфор)", "P (phosphorus)"],
    "Se": ["Se (селен)", "Se (селен)", "Se (selenium)"],
    "Si": ["Si (кремний)", "Si (кремний)", "Si (silicon)"],
    "Zn": ["Zn (мыс)", "Zn (цинк)", "Zn (zinc)"],
    "predict_button": ["Болжам жасау", "Сделать прогноз", "Predict"],
    "models_load_success": ["Модельдер репозиторийден жүктелді.", "Модели загружены из репозитория.", "Models loaded from repository."],
    "upload_models_now": ["Модельдерді қазір жүктеңіз", "Загрузить модели сейчас", "Upload models now"],
    "missing_cols": [
        "Файлда қажетті бағандар жоқ: ",
        "В файле отсутствуют необходимые столбцы: ",
        "File is missing required columns: "
    ],
    "download_button": ["Алдын ала болжамдарды жүктеу", "Скачать прогнозы", "Download predictions"],
    "height_1y": ["1 жылдан кейінгі бой (см)", "Рост через 1 год (см)", "Height after 1 year (cm)"],
    "height_2y": ["2 жылдан кейінгі бой (см)", "Рост через 2 года (см)", "Height after 2 years (cm)"],
    "weight_1y": ["1 жылдан кейінгі салмақ (кг)", "Вес через 1 год (кг)", "Weight after 1 year (kg)"],
    "weight_2y": ["2 жылдан кейінгі салмақ (кг)", "Вес через 2 года (кг)", "Weight after 2 years (kg)"],
    "growth_curve": ["Өсу қисығы", "Кривая роста", "Growth curve"],
    "app_caption": [
        "Бұл қосымша ChatGPT арқылы жасалған — өзгерістер сұраңыз.",
        "Приложение сгенерировано ChatGPT — просите изменения по желанию.",
        "App generated by ChatGPT — ask for customizations."
    ],
    "error_load_model": [
        "Модель жүктелмеді: ",
        "Ошибка загрузки модели: ",
        "Model load failed: "
    ],
    "choose_action": ["Таңдау:", "Выбрать:", "Choose:"],
    "menu_single": ["Жеке болжам", "Одиночный прогноз", "Single prediction"],
    "menu_batch": ["Көтерме болжам", "Пакетный прогноз", "Batch prediction"]
}

# Helper to select language index
if "lang_idx" not in st.session_state:
    st.session_state["lang_idx"] = 0  # default Kazakh

# Language selector at top
col_lang, _ = st.columns([1, 5])
with col_lang:
    st.session_state["lang_idx"] = st.selectbox(
        "",
        options=[TXT["lang_names"][0], TXT["lang_names"][1], TXT["lang_names"][2]],
        index=0
    )
# Map selected label back to index
lang_label = st.session_state["lang_idx"]
lang_idx = 0
if lang_label == TXT["lang_names"][1]:
    lang_idx = 1
elif lang_label == TXT["lang_names"][2]:
    lang_idx = 2
else:
    lang_idx = 0

# translation helper
def t(key):
    val = TXT.get(key)
    if val is None:
        return ""
    return val[lang_idx]

# -------- SAFE MODEL LOADING (no crash) ----------
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        # show small message in the selected language
        st.sidebar.warning(f"{t('error_load_model')}{path} — {e}")
        return None

# Attempt to load from repo root (app.py assumed in repo root)
model_h = safe_load("model_height.joblib")
model_h_2y = safe_load("model_height_2y.joblib")
model_w = safe_load("model_weight.joblib")
model_w_2y = safe_load("model_weight_2y.joblib")
scaler = safe_load("scaler.joblib")

models_loaded = all([model_h is not None, model_h_2y is not None, model_w is not None, model_w_2y is not None, scaler is not None])

# HEADER
st.title(t("title"))
st.write(t("subtitle"))

if models_loaded:
    st.success(t("models_loaded"))
else:
    st.error(t("models_not_loaded"))

# UPLOAD AREA (optional)
with st.expander(t("upload_expander")):
    st.write(t("upload_hint"))
    up_mh = st.file_uploader("model_height.joblib", type=["joblib","pkl"], key="up_mh")
    up_mh2 = st.file_uploader("model_height_2y.joblib", type=["joblib","pkl"], key="up_mh2")
    up_mw = st.file_uploader("model_weight.joblib", type=["joblib","pkl"], key="up_mw")
    up_mw2 = st.file_uploader("model_weight_2y.joblib", type=["joblib","pkl"], key="up_mw2")
    up_sc = st.file_uploader("scaler.joblib", type=["joblib","pkl"], key="up_sc")

    if st.button(t("upload_models_now")):
        try:
            if up_mh and up_mh2 and up_mw and up_mw2 and up_sc:
                model_h = joblib.load(up_mh)
                model_h_2y = joblib.load(up_mh2)
                model_w = joblib.load(up_mw)
                model_w_2y = joblib.load(up_mw2)
                scaler = joblib.load(up_sc)
                models_loaded = True
                st.success(t("models_load_success"))
            else:
                st.warning(t("upload_hint"))
        except Exception as e:
            st.error(f"{t('error_load_model')}{e}")

# Sidebar menu in chosen language
st.sidebar.header(t("choose_action"))
menu_option = st.sidebar.radio("", (t("menu_single"), t("menu_batch")))

# SINGLE PREDICTION
if menu_option == t("menu_single"):
    st.header(t("single_pred"))

    age = st.number_input(t("age"), min_value=0, max_value=18, value=5)
    # gender options text depends on language; map to numeric for model
    gender_display = t("gender_options")
    g_opts = None
    if lang_idx == 0:  # Kazakh simple
        g_opts = TXT["gender_options"][0]
    elif lang_idx == 1:  # Russian
        g_opts = TXT["gender_options"][1]
    else:
        g_opts = TXT["gender_options"][2]

    gender_sel = st.selectbox(t("gender"), options=g_opts)
    # map display back to model values: Boy -> 1, Girl -> 0
    # we will accept either local words; map by index
    gender_num = 1 if gender_sel == g_opts[0] else 0

    height = st.number_input(t("height"), min_value=30.0, max_value=220.0, value=110.0)
    weight = st.number_input(t("weight"), min_value=1.0, max_value=200.0, value=17.0)

    B = st.number_input(t("B"), value=1.23)
    Mg = st.number_input(t("Mg"), value=72.76)
    P = st.number_input(t("P"), value=171.17)
    Se = st.number_input(t("Se"), value=0.52)
    Si = st.number_input(t("Si"), value=19.8)
    Zn = st.number_input(t("Zn"), value=179.6)

    if st.button(t("predict_button")):
        if not models_loaded:
            st.error(t("models_not_loaded"))
        else:
            BMI = weight / (height/100)**2
            df = pd.DataFrame([[age, gender_num, height, weight, BMI, B, Mg, P, Se, Si, Zn]],
                              columns=FEATURE_NAMES)
            try:
                Xs = scaler.transform(df)
                h1 = float(model_h.predict(Xs)[0])
                h2 = float(model_h_2y.predict(Xs)[0])
                w1 = float(model_w.predict(Xs)[0])
                w2 = float(model_w_2y.predict(Xs)[0])

                st.metric(t("height_1y"), f"{h1:.2f} cm")
                st.metric(t("height_2y"), f"{h2:.2f} cm")
                st.metric(t("weight_1y"), f"{w1:.2f} kg")
                st.metric(t("weight_2y"), f"{w2:.2f} kg")

                # plot growth curves
                years = [0, 1, 2]
                heights = [height, h1, h2]
                weights = [weight, w1, w2]

                fig, ax = plt.subplots()
                ax.plot(years, heights, marker='o')
                ax.set_title(t("growth_curve"))
                ax.set_xticks(years)
                ax.set_ylabel(t("height"))
                st.pyplot(fig)

                fig2, ax2 = plt.subplots()
                ax2.plot(years, weights, marker='o')
                ax2.set_title(t("growth_curve"))
                ax2.set_xticks(years)
                ax2.set_ylabel(t("weight"))
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"{t('error_load_model')}{e}")

# BATCH PREDICTION
else:
    st.header(t("batch_pred"))
    st.write({
        0: "CSV немесе XLSX файл жүктеңіз. Файлда бағанда болуы керек: age, height, weight, B, Mg, P, Se, Si, Zn",
        1: "Загрузите CSV или XLSX. В файле должны быть столбцы: age, height, weight, B, Mg, P, Se, Si, Zn",
        2: "Upload CSV or XLSX. File must contain columns: age, height, weight, B, Mg, P, Se, Si, Zn"
    }[lang_idx])

    uploaded = st.file_uploader("", type=["csv", "xlsx"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"{t('error_load_model')}{e}")
            df = None

        if df is not None:
            required = ['age','height','weight','B','Mg','P','Se','Si','Zn']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(t("missing_cols") + ", ".join(missing))
            elif not models_loaded:
                st.error(t("models_not_loaded"))
            else:
                # map gender if exists
                if 'gender' in df.columns:
                    # If gender values are words, try to map common ones; default 1/0
                    df['gender_num'] = df['gender'].map({'T':1,'F':0,'Boy':1,'Girl':0,'Мальчик':1,'Девочка':0,'Ер':1,'Қыз':0}).fillna(0)
                if 'BMI' not in df.columns:
                    df['BMI'] = df['weight'] / (df['height']/100)**2

                X = df[FEATURE_NAMES]
                try:
                    Xs = scaler.transform(X)
                    df["height_1y"] = model_h.predict(Xs)
                    df["height_2y"] = model_h_2y.predict(Xs)
                    df["weight_1y"] = model_w.predict(Xs)
                    df["weight_2y"] = model_w_2y.predict(Xs)

                    st.dataframe(df.head())

                    out = io.BytesIO()
                    df.to_excel(out, index=False)
                    out.seek(0)
                    st.download_button(t("download_button"), data=out, file_name="Predictions.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.error(f"{t('error_load_model')}{e}")

st.caption(t("app_caption"))
