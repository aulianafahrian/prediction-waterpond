import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_pymongo import PyMongo
import joblib
from dataset_ftr import create_features
from datetime import datetime, timedelta
import random
import os

app = Flask(__name__)

# Konfigurasi MongoDB
app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
mongo = PyMongo(app)

# Load model
model_ph = joblib.load("model_water_ph_rf.joblib")
model_tds = joblib.load("model_tds_rf.joblib")
model_suhu = joblib.load("model_water_temp_rf.joblib")
model_klasifikasi = joblib.load("model_klasifikasi.joblib")

latest_data = {}  # untuk realtime card
alerts = []  # Untuk data hasil prediksi

@app.route("/", methods=["GET", "POST"])
def index():
    global latest_data, alerts  # Deklarasi sebagai global

    # Ambil data historis terbaru untuk ditampilkan pada card
    latest_sensor_data = mongo.db.sensor_data.find().sort('create_date', -1).limit(1)
    
    # Ambil data terakhir untuk card
    if latest_sensor_data:
        latest_sensor = latest_sensor_data[0]
        latest_data = {  # Assign data terbaru ke latest_data
            "ph": round(latest_sensor['water_ph'], 2),
            "tds": round(latest_sensor['tds'], 2),
            "suhu": round(latest_sensor['water_temp'], 2)
        }

    # Simulasi Data Dummy untuk Testing
    ph_input = round(random.uniform(5.0, 8.0), 2)  # pH antara 5 dan 8
    tds_input = round(random.uniform(100.0, 500.0), 2)  # TDS antara 100 dan 500
    suhu_input = round(random.uniform(18.0, 30.0), 2)  # Suhu antara 18 dan 30

    # Simpan latest untuk dashboard realtime
    latest_data = {
        "ph": ph_input,
        "tds": tds_input,
        "suhu": suhu_input
    }

    # Simpan data dummy ke MongoDB dan CSV
    now = datetime.now()
    next_time = next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Menyimpan data ke MongoDB
    sensor_data = {
        'create_date': next_time,
        'water_ph': ph_input,
        'tds': tds_input,
        'water_temp': suhu_input
    }
    mongo.db.sensor_data.insert_one(sensor_data)  # Menyimpan ke MongoDB

    # Menyimpan data ke CSV
    df_cleaned = pd.read_csv("data_cleaned.csv")
    new_row = {
        'created_date': next_time,
        'water_pH': ph_input,
        'TDS': tds_input,
        'water_temp': suhu_input
    }
    df_cleaned = pd.concat([df_cleaned, pd.DataFrame([new_row])], ignore_index=True)
    df_cleaned.to_csv("data_cleaned.csv", index=False)

    # Ambil 48 jam historis untuk fitur waktu
    df_cleaned['created_date'] = pd.to_datetime(df_cleaned['created_date'])
    context = df_cleaned.sort_values("created_date").tail(48).copy()

    # Forecast 12 jam ke depan
    current_history = context.copy()
    N_FORECAST = 12

    # Reset alerts untuk hanya menyimpan hasil prediksi terbaru
    alerts = []  # Clear previous predictions

    for i in range(N_FORECAST):
        df_feat = create_features(current_history)
        df_feat = df_feat.dropna()

        FEATURES = [col for col in df_feat.columns if col not in ['water_pH', 'TDS', 'water_temp']]
        last_row_features = df_feat.iloc[[-1]][FEATURES]

        # Prediksi
        ph_pred = model_ph.predict(last_row_features)[0]
        tds_pred = model_tds.predict(last_row_features)[0]
        suhu_pred = model_suhu.predict(last_row_features)[0]

        pred_time = current_history['created_date'].iloc[-1] + timedelta(hours=1)

        # Klasifikasi
        fitur_klas = pd.DataFrame([[ph_pred, tds_pred, suhu_pred]],
                                  columns=['water_pH', 'TDS', 'water_temp'])
        klasifikasi = model_klasifikasi.predict(fitur_klas)[0]
        prob = model_klasifikasi.predict_proba(fitur_klas)[0][1]
        status = "PERLU GANTI AIR" if klasifikasi == 1 else "AMAN"

        # Simpan hasil prediksi terbaru
        alerts.append({
            "jam_ke": i + 1,
            "waktu": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ph": round(ph_pred, 2),
            "tds": round(tds_pred, 2),
            "suhu": round(suhu_pred, 2),
            "status": status,
            "prob": f"{prob:.2f}"
        })

        # Tambah ke histori
        next_row = {
            'created_date': pred_time,
            'water_pH': ph_pred,
            'TDS': tds_pred,
            'water_temp': suhu_pred
        }
        current_history = pd.concat([current_history, pd.DataFrame([next_row])], ignore_index=True)

    return render_template("index.html", alerts=alerts, latest=latest_data)



# Endpoint untuk card realtime (dipanggil dari JavaScript tiap 5 detik)
@app.route("/api/latest")
def api_latest():
    if latest_data:
        return jsonify(latest_data)
    else:
        return jsonify({"error": "Data belum tersedia"})


# Endpoint untuk memberikan data prediksi (diakses oleh frontend untuk update tabel)
@app.route("/api/predictions")
def api_predictions():
    return jsonify(alerts)  # Mengembalikan data prediksi yang ada di alerts


# Endpoint tambahan jika ingin simulasi dari script eksternal
@app.route("/api/post", methods=["POST"])
def api_post():
    global latest_data
    try:
        data = request.json
        latest_data = {
            "ph": round(float(data.get("ph", 0)), 2),
            "tds": round(float(data.get("tds", 0)), 2),
            "suhu": round(float(data.get("suhu", 0)), 2)
        }
        return jsonify({"success": True, "message": "Data diterima"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
