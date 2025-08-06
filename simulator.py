import requests
import random
import time

# URL API Flask lokal
URL = "http://127.0.0.1:5000/api/post"

def generate_data():
    return {
        "ph": round(random.uniform(6.5, 8.5), 2),
        "tds": round(random.uniform(150, 400), 2),
        "suhu": round(random.uniform(24, 30), 2)
    }

def send_data():
    data = generate_data()
    try:
        response = requests.post(URL, json=data)
        print(f"✅ Data dikirim: {data} | Status: {response.status_code} | Balasan: {response.json()}")
    except Exception as e:
        print(f"❌ Gagal kirim data: {e}")

if __name__ == "__main__":
    print("🚀 Memulai simulasi pengiriman data...")
    while True:
        send_data()
        time.sleep(3600)  # ubah ke 3600 untuk simulasi tiap 1 jam
