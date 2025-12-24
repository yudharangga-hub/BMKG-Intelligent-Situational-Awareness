import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import json

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'arsip_scraping_lengkap.csv')
OUTPUT_JSON = os.path.join(BASE_DIR, 'static', 'data_map.json')

def main():
    print("="*60)
    print("🗺️ GENERATING GEO-MAP DATA WITH CONTEXTUAL SAMPLES")
    print("="*60)

    device = 0 if torch.cuda.is_available() else -1
    
    # 1. Load Data
    print("📂 Loading data...")
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=['Komentar'], inplace=True)
    df = df.head(2000) # Ambil 2000 data terbaru
    
    texts = df['Komentar'].tolist()

    # 2. NER Pipeline
    print("🧠 Memuat Model BERT NER...")
    # Gunakan model yang ringan tapi cepat untuk demo
    ner_pipeline = pipeline("ner", model="cahya/bert-base-indonesian-ner", device=device, aggregation_strategy="simple")

    # 3. Ekstraksi Lokasi & Mapping Komentar
    print("🕵️‍♂️ Mendeteksi lokasi dan mengaitkan komentar...")
    
    # Dictionary untuk menyimpan: {'Jakarta': {'count': 5, 'samples': ['komentar 1', 'komentar 2']}}
    loc_data = {} 
    
    blacklist = ["bmkg", "indonesia", "aplikasi", "info", "gempa", "cuaca", "lokasi", "daerah", "barusan", "pusat", "selatan", "utara", "barat", "timur"]

    for text in tqdm(texts):
        if len(str(text)) < 10: continue
        
        try:
            results = ner_pipeline(text)
            for entity in results:
                if entity['entity_group'] in ['LOC', 'GPE']:
                    word = entity['word'].strip()
                    
                    # Filter kata pendek/blacklist
                    if word.lower() not in blacklist and len(word) > 3:
                        # Normalisasi nama lokasi (Title Case)
                        loc_name = word.title()
                        
                        if loc_name not in loc_data:
                            loc_data[loc_name] = {'count': 0, 'samples': []}
                        
                        loc_data[loc_name]['count'] += 1
                        
                        # Simpan max 3 komentar unik per lokasi untuk sampel
                        if len(loc_data[loc_name]['samples']) < 3:
                            # Bersihkan teks sedikit agar rapi di popup
                            clean_snippet = text.replace('"', '').replace("'", "")[:100] + "..."
                            if clean_snippet not in loc_data[loc_name]['samples']:
                                loc_data[loc_name]['samples'].append(clean_snippet)
        except: continue

    # Ambil Top 50 Lokasi terbanyak disebut
    sorted_locs = sorted(loc_data.items(), key=lambda x: x[1]['count'], reverse=True)[:50]
    print(f"\n📍 {len(sorted_locs)} Lokasi Signifikan Terdeteksi.")

    # 4. Geocoding
    print("\n🌍 Mengambil Koordinat GPS...")
    geolocator = Nominatim(user_agent="bmkg_intel_thesis_v3", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

    final_map_data = []
    
    for loc_name, data in tqdm(sorted_locs):
        try:
            location = geocode(f"{loc_name}, Indonesia")
            if location:
                final_map_data.append({
                    "name": loc_name,
                    "lat": location.latitude,
                    "lon": location.longitude,
                    "count": data['count'],
                    "samples": data['samples'] # <--- INI TAMBAHAN PENTINGNYA
                })
        except Exception as e:
            print(f"⚠️ Skip '{loc_name}': {e}")

    # 5. Simpan
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_map_data, f, indent=4)

    print(f"\n✅ Data Peta Siap: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()