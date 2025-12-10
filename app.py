import joblib

models = {
  'h1': joblib.load("model_height.joblib"),
  'h2': joblib.load("model_height_2y.joblib"),
  'w1': joblib.load("model_weight.joblib"),
  'w2': joblib.load("model_weight_2y.joblib")
}
scaler = joblib.load("scaler.joblib")
