from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # align input with model features
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(df)
    return {"predicted_wait_time_minutes": float(prediction[0])}

