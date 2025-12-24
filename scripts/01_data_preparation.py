import os
import re
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import sys

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'arsip_scraping_lengkap.csv')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_absa_labeled.csv')

def clean_text(text):
    """Membersihkan teks dari URL, tanda baca, dan spasi berlebih."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_sentiment(star):
    """Mapping Bintang ke Label Sentimen"""
    if star <= 2: return 'Negatif'
    elif star == 3: return 'Netral'
    else: return 'Positif'

def main():
    print("="*60)
    print("🚀 MEMULAI DATA PREPARATION & AUTO-LABELING (FIXED)")
    print(f"📂 Membaca data dari: {DATA_RAW}")
    print("="*60)

    # 1. Cek GPU
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Terdeteksi: {gpu_name} (Ready for Heavy Lifting!)")
    else:
        print("⚠️ PERINGATAN: GPU tidak terdeteksi! Proses akan sangat lambat.")
        device = -1

    # 2. Load Data
    try:
        df = pd.read_csv(DATA_RAW)
        print(f"📊 Total Data Awal: {len(df)} baris")
    except FileNotFoundError:
        print("❌ Error: File CSV tidak ditemukan.")
        sys.exit()

    # 3. Cleaning & Filtering
    print("🧹 Membersihkan teks...")
    df.dropna(subset=['Komentar'], inplace=True)
    df.drop_duplicates(subset=['Komentar'], inplace=True)
    
    # Terapkan cleaning
    df['clean_text'] = df['Komentar'].apply(clean_text)
    
    # Hapus data kosong hasil cleaning
    initial_count = len(df)
    df = df[df['clean_text'].astype(bool)]
    df = df[df['clean_text'].str.strip() != '']
    dropped_count = initial_count - len(df)
    
    print(f"🗑️ Dibuang {dropped_count} data sampah (hanya emoji/tanda baca).")
    print(f"✅ Sisa Data Bersih: {len(df)} baris")
    
    # Mapping Sentimen
    df['Sentimen'] = df['Bintang'].apply(map_sentiment)

    # 4. Auto-Labeling
    print("\n🧠 Memuat Model Zero-Shot (XLM-Roberta)...")
    classifier = pipeline("zero-shot-classification", 
                          model="joeddav/xlm-roberta-large-xnli", 
                          device=device) 

    candidate_labels = ["akurasi cuaca", "tampilan aplikasi", "kinerja aplikasi lambat"]
    label_map = {
        "akurasi cuaca": "Akurasi",
        "tampilan aplikasi": "UI/UX",
        "kinerja aplikasi lambat": "Performa"
    }

    print("⚡ Memulai Auto-Labeling pada RTX 3080...")
    aspects = []
    confidences = []
    
    # Batch processing
    batch_size = 32
    comments = df['clean_text'].tolist()

    # Loop Processing
    for i in tqdm(range(0, len(comments), batch_size), desc="Processing Batches"):
        batch = comments[i:i+batch_size]
        
        if not batch: continue
            
        try:
            results = classifier(batch, candidate_labels, multi_label=False)
            
            for res in results:
                top_label = res['labels'][0]
                top_score = res['scores'][0]
                aspects.append(label_map[top_label])
                confidences.append(top_score)
                
        except Exception as e:
            print(f"\n⚠️ Error pada batch {i}: {e}")
            # Fallback jika error
            for _ in batch:
                aspects.append("Lainnya")
                confidences.append(0.0)

    # Final Check
    if len(aspects) == len(df):
        df['Aspek_Terdeteksi'] = aspects
        df['Confidence_Score'] = confidences
        
        # 5. Saving
        df.to_csv(DATA_PROCESSED, index=False)
        print("\n" + "="*60)
        print(f"✅ SELESAI! Data tersimpan di: {DATA_PROCESSED}")
        print("Sampel Data:")
        print(df[['Komentar', 'Aspek_Terdeteksi', 'Sentimen']].head())
        print("="*60)
    else:
        print("❌ Terjadi ketidakcocokan jumlah data.")

if __name__ == "__main__":
    main()