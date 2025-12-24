import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import re

# Gunakan backend non-interaktif agar tidak error di server
import matplotlib
matplotlib.use('Agg')

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'arsip_scraping_lengkap.csv')
OUTPUT_IMG = os.path.join(BASE_DIR, 'static', 'images', 'wordcloud_freq.png')

# Stopwords Bahasa Indonesia (Manual agar ringan)
STOPWORDS = set([
    'dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu', 'ada', 'untuk', 'tapi', 
    'saya', 'tidak', 'gak', 'bisa', 'aplikasi', 'bmkg', 'nya', 'sangat', 
    'terlalu', 'tolong', 'min', 'kalau', 'pas', 'yg', 'ga', 'aja', 'juga', 
    'mau', 'sudah', 'lagi', 'kalo', 'sama', 'bikin', 'malah', 'kok', 'biar',
    'karena', 'banget', 'apk', 'info', 'update', 'padahal', 'lebih', 'masih'
])

def clean_text(text):
    """Membersihkan teks untuk WordCloud"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Hapus angka & simbol
    return text

def main():
    print("☁️ GENERATING WORDCLOUD...")
    
    if not os.path.exists(DATA_PATH):
        print("❌ CSV Not Found!")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} comments.")

    # 2. Gabungkan seluruh komentar jadi satu string raksasa
    all_text = " ".join(df['Komentar'].apply(clean_text))

    # 3. Generate WordCloud
    wc = WordCloud(
        width=1600, 
        height=800,
        background_color='white',
        stopwords=STOPWORDS,
        colormap='ocean', # Tema Biru Laut/Langit (Sesuai BMKG)
        min_font_size=10,
        max_words=200
    ).generate(all_text)

    # 4. Simpan Gambar
    plt.figure(figsize=(20,10), facecolor=None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Pastikan folder images ada
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    
    plt.savefig(OUTPUT_IMG)
    plt.close()
    
    print(f"✅ WordCloud Saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()