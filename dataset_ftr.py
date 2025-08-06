import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("Memulai proses pembuatan dataset dengan fitur...")

# --- Memuat Data Asli ---
try:
    file_path = 'data_cleaned.csv'
    df = pd.read_csv(file_path)
    print(f"Data '{file_path}' berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
    exit()

# --- Fungsi untuk Membuat Fitur ---
def create_features(df_input):
    """Membuat fitur time series dari dataframe."""
    df_feat = df_input.copy()
    df_feat['created_date'] = pd.to_datetime(df_feat['created_date'])
    df_feat = df_feat.set_index('created_date')
    
    # Fitur berbasis waktu
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_year'] = df_feat.index.dayofyear
    df_feat['month'] = df_feat.index.month
    
    # Fitur lag
    target_cols = ['water_pH', 'TDS', 'water_temp']
    for col in target_cols:
        for i in [1, 2, 3, 24]:
            df_feat[f'{col}_lag_{i}hr'] = df_feat[col].shift(i)
            
    # Fitur window (rolling)
    for col in target_cols:
        df_feat[f'{col}_roll_mean_3hr'] = df_feat[col].rolling(window=3).mean()
        df_feat[f'{col}_roll_mean_24hr'] = df_feat[col].rolling(window=24).mean()
        
    return df_feat

# --- Menerapkan Fungsi dan Menyimpan Hasil ---
df_with_features = create_features(df)

# Menghapus baris yang mengandung NaN (muncul akibat proses lagging/rolling)
df_final = df_with_features.dropna()
print("Rekayasa fitur selesai dan baris NaN telah dihapus.")

# Menyimpan ke file CSV baru
output_filename = 'data_with_features.csv'
df_final.to_csv(output_filename)

print(f"\nProses selesai. Dataset dengan fitur lengkap telah disimpan sebagai '{output_filename}'")