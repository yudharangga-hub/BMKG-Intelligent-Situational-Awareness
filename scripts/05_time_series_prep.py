import pandas as pd
import os
import json

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_absa_labeled.csv')
OUTPUT_JSON = os.path.join(BASE_DIR, 'static', 'trends_data.json')

def main():
    print("="*60)
    print("📈 MEMULAI TIME SERIES ANALYSIS (AGGREGATION)")
    print("="*60)

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # 2. Convert Tanggal
    print("📅 Memproses kolom tanggal...")
    # Pastikan format tanggal sesuai data Anda
    # Jika ada error parsing, errors='coerce' akan mengubahnya jadi NaT (Not a Time) lalu kita buang
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df.dropna(subset=['Tanggal'], inplace=True)
    
    # Urutkan berdasarkan tanggal
    df.sort_values('Tanggal', inplace=True)
    
    print(f"   Rentang Data: {df['Tanggal'].min()} s.d {df['Tanggal'].max()}")

    # 3. Resampling Harian (Daily Trend)
    # Kita hitung jumlah sentimen Positif, Negatif, Netral per Hari
    print("📊 Menghitung tren sentimen harian...")
    
    # Buat kolom dummy untuk perhitungan
    df['count'] = 1
    
    # Pivot Table: Index=Tanggal, Columns=Sentimen, Values=Count
    daily_sentiment = df.pivot_table(
        index=df['Tanggal'].dt.date, 
        columns='Sentimen', 
        values='count', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Pivot Table: Index=Tanggal, Columns=Aspek, Values=Count
    daily_aspect = df.pivot_table(
        index=df['Tanggal'].dt.date, 
        columns='Aspek_Terdeteksi', 
        values='count', 
        aggfunc='sum',
        fill_value=0
    )

    # 4. Format ke JSON (Agar mudah dibaca Chart.js di Frontend)
    dates = [str(d) for d in daily_sentiment.index]
    
    trends_data = {
        "dates": dates,
        "sentiment": {
            "positif": daily_sentiment.get('Positif', pd.Series(0, index=daily_sentiment.index)).tolist(),
            "negatif": daily_sentiment.get('Negatif', pd.Series(0, index=daily_sentiment.index)).tolist(),
            "netral": daily_sentiment.get('Netral', pd.Series(0, index=daily_sentiment.index)).tolist(),
        },
        "aspect": {
            "akurasi": daily_aspect.get('Akurasi', pd.Series(0, index=daily_aspect.index)).tolist(),
            "ui_ux": daily_aspect.get('UI/UX', pd.Series(0, index=daily_aspect.index)).tolist(),
            "performa": daily_aspect.get('Performa', pd.Series(0, index=daily_aspect.index)).tolist()
        }
    }

    # 5. Simpan
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(trends_data, f, indent=4)

    print(f"\n✅ Data Tren Disimpan: {OUTPUT_JSON}")
    print("Sample Data:")
    print(f"   Dates: {trends_data['dates'][:3]} ...")
    print(f"   Positif: {trends_data['sentiment']['positif'][:3]} ...")

if __name__ == "__main__":
    main()