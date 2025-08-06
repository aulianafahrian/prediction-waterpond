# File: api/index.py

import pandas as pd
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
import joblib
from dataset_ftr import create_features
from datetime import datetime, timedelta
import random
import os

app = Flask(__name__)

# Konfigurasi MongoDB dari environment (agar aman di Vercel)
app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
mongo = PyMongo(app)

# Load model (gunakan path relatif dari root)
model_ph = joblib.load("model_water_ph_rf.joblib")
model_tds = joblib.load("model_tds_rf.joblib")
model_suhu = joblib.load("model_water_temp_rf.joblib")
model_klasifikasi = joblib.load("model_klasifikasi.joblib")

latest_data = {}
alerts = []

@app.route("/api/latest")
def api_latest():
    if latest_data:
        return jsonify(latest_data)
    else:
        return jsonify({"error": "Data belum tersedia"})

@app.route("/api/predictions")
def api_predictions():
    return jsonify(alerts)

@app.route("/api/predict", methods=["POST"])
def predict():
    global latest_data, alerts

    now = datetime.now()
    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    ph_input = round(random.uniform(5.0, 8.0), 2)
    tds_input = round(random.uniform(100.0, 500.0), 2)
    suhu_input = round(random.uniform(18.0, 30.0), 2)

    latest_data = {
        "ph": ph_input,
        "tds": tds_input,
        "suhu": suhu_input
    }

    sensor_data = {
        'create_date': next_time,
        'water_ph': ph_input,
        'tds': tds_input,
        'water_temp': suhu_input
    }
    mongo.db.sensor_data.insert_one(sensor_data)

    df_cleaned = pd.read_csv("data_cleaned.csv")
    new_row = {
        'created_date': next_time,
        'water_pH': ph_input,
        'TDS': tds_input,
        'water_temp': suhu_input
    }
    df_cleaned = pd.concat([df_cleaned, pd.DataFrame([new_row])], ignore_index=True)
    df_cleaned.to_csv("data_cleaned.csv", index=False)

    df_cleaned['created_date'] = pd.to_datetime(df_cleaned['created_date'])
    context = df_cleaned.sort_values("created_date").tail(48).copy()
    current_history = context.copy()
    N_FORECAST = 12
    alerts.clear()

    for i in range(N_FORECAST):
        df_feat = create_features(current_history)
        df_feat = df_feat.dropna()
        FEATURES = [col for col in df_feat.columns if col not in ['water_pH', 'TDS', 'water_temp']]
        last_row_features = df_feat.iloc[[-1]][FEATURES]

        ph_pred = model_ph.predict(last_row_features)[0]
        tds_pred = model_tds.predict(last_row_features)[0]
        suhu_pred = model_suhu.predict(last_row_features)[0]
        pred_time = current_history['created_date'].iloc[-1] + timedelta(hours=1)

        fitur_klas = pd.DataFrame([[ph_pred, tds_pred, suhu_pred]],
                                  columns=['water_pH', 'TDS', 'water_temp'])
        klasifikasi = model_klasifikasi.predict(fitur_klas)[0]
        prob = model_klasifikasi.predict_proba(fitur_klas)[0][1]
        status = "PERLU GANTI AIR" if klasifikasi == 1 else "AMAN"

        alerts.append({
            "jam_ke": i + 1,
            "waktu": pred_time.strftime("%Y-%m-%d %H:%M"),
            "ph": round(ph_pred, 2),
            "tds": round(tds_pred, 2),
            "suhu": round(suhu_pred, 2),
            "status": status,
            "prob": f"{prob:.2f}"
        })

        next_row = {
            'created_date': pred_time,
            'water_pH': ph_pred,
            'TDS': tds_pred,
            'water_temp': suhu_pred
        }
        current_history = pd.concat([current_history, pd.DataFrame([next_row])], ignore_index=True)

    return jsonify({"message": "Prediksi berhasil dilakukan", "alerts": alerts})
