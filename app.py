import joblib

models = {
  'h1': joblib.load("models/model_height.joblib"),
  'h2': joblib.load("models/model_height_2y.joblib"),
  'w1': joblib.load("models/model_weight.joblib"),
  'w2': joblib.load("models/model_weight_2y.joblib")
}
scaler = joblib.load("models/scaler.joblib")
