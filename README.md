# BMKG Intelligent Situational Awareness

Sistem analitik cerdas untuk monitoring, analisis, dan visualisasi data cuaca, gempa, dan sentimen berbasis NLP dan Word2Vec. Mendukung chatbot BMKG, smart reply, dan Semantic Lab (Kamus Slang).

---

## Fitur Utama
- **Chatbot BMKG**: Jawab pertanyaan cuaca, gempa, dan peringatan secara real-time.
- **Smart Reply**: Rekomendasi balasan cerdas untuk pengelola aplikasi.
- **Semantic Lab**: Eksplorasi relasi kata, sinonim, dan jaringan makna berbasis Word2Vec.
- **Visualisasi Jaringan Kata**: Interaktif, berbasis vis-network.
- **Analisis Sentimen**: Klasifikasi sentimen otomatis dari data keluhan.
- **Integrasi Data BMKG**: Live feed cuaca, gempa, dan peringatan dini.

---

## Struktur Folder

```
BMKG_NLP_Analytics/
├── app.py
├── config.py
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   ├── dataset_absa_labeled.csv
│   │   └── dataset_emotion_labeled.csv
│   └── raw/
│       └── arsip_scraping_lengkap.csv
├── models/
│   ├── aspect_model/
│   │   └── ... (model IndoBERT, tokenizer, dsb)
│   ├── emotion_model/
│   │   └── ... (model IndoBERT, tokenizer, dsb)
│   ├── word2vec/
│   │   └── word2vec.bin
│   ├── checkpoints/
│   │   └── ... (checkpoint training aspect)
│   └── emotion_checkpoints/
│       └── ... (checkpoint training emotion)
├── notebooks/
├── screenshots/
├── scripts/
│   ├── 01_data_preparation.py
│   ├── 02_train_aspect_model.py
│   ├── 03_ner_geomapping.py
│   ├── 04_emotion_training.py
│   ├── 05_time_series_prep.py
│   ├── 06_generate_metrics.py
│   ├── 07_bug_extraction.py
│   ├── 09_generate_wordcloud.py
│   ├── 10_run_benchmark.py
│   └── train_word2vec_from_csv.py
├── static/
│   ├── bug_report.json
│   ├── data_map.json
│   ├── model_metrics.json
│   ├── trends_data.json
│   ├── css/
│   │   └── style.css
│   ├── images/
│   ├── js/
│   │   ├── chatbot_widget.js
│   │   └── semantic_lab.js
├── templates/
│   ├── base.html
│   ├── dev_dashboard.html
│   ├── index.html
│   ├── ner_map.html
│   └── trends.html
│   └── semantic_lab.html
├── utils/
│   ├── __init__.py
│   ├── bmkg_api.py
│   ├── model_handler.py
│   ├── preprocessing.py
│   └── word2vec_handler.py
└── ...
```

---

## Cara Menjalankan

1. **Install dependensi**
    ```
    pip install -r requirements.txt
    ```
2. **Training Word2Vec (opsional, jika ingin update model):**
    ```
    python scripts/train_word2vec_from_csv.py
    ```
3. **Jalankan aplikasi**
    ```
    python app.py
    ```
4. **Akses di browser**
    - Dashboard: http://127.0.0.1:5000/
    - Semantic Lab: http://127.0.0.1:5000/semantic_lab

---

## Catatan
- Model IndoBERT dan Word2Vec tidak disertakan di repo (file besar), silakan generate sendiri dari data.
- Untuk visualisasi jaringan kata, gunakan browser modern (sudah terintegrasi vis-network).
- Semua API dan fitur utama sudah terdaftar di app.py.

---

## Kontribusi
Pull request dan issue sangat terbuka untuk pengembangan lebih lanjut!

---

## Lisensi
MIT
# 🌍 Intelligent Situational Awareness Platform (BMKG-INTEL)

> **Thesis Project: Magister Teknik Informatika**
>
> *Rancang Bangun Sistem Analisis Sentimen & Pemetaan Geospasial untuk Aplikasi Info BMKG menggunakan Hybrid NLP (IndoBERT & NER) dan Integrasi Data Kebencanaan Real-Time.*

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-Flask-green?style=for-the-badge&logo=flask&logoColor=white)
![AI Model](https://img.shields.io/badge/Model-IndoBERT%20Transformer-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Prototype-yellow?style=for-the-badge)

---

## 📸 Galeri Sistem (System Preview)

Berikut adalah tampilan antarmuka dari sistem yang telah dikembangkan:

### 1. Dashboard Utama (Analisis Sentimen & Integrasi API)
*Memproses input teks pengguna menggunakan IndoBERT dan menampilkan data Gempa/Cuaca Real-time.*
![Dashboard View](screenshots/Dashboard.jpg)

### 2. Geo-Spatial Intelligence (Peta Sebaran)
*Validasi silang antara lokasi Laporan Warga (Oranye) dan Sensor Gempa (Merah) menggunakan NER & LeafletJS.*
![GeoMap View](screenshots/GeoMap.jpg)

### 3. Developer Action Center
*Prescriptive Analytics yang memberikan rekomendasi perbaikan teknis berdasarkan klaster keluhan.*
![Dev Center View](screenshots/DevCenter.jpg)

---

## 📖 Latar Belakang

Ribuan ulasan membanjiri aplikasi **Info BMKG** setiap harinya. Analisis manual terhadap "Big Data" ini tidak efisien dan sering melewatkan laporan kritis (seperti bug notifikasi gempa). Selain itu, tidak ada korelasi spasial untuk memvalidasi apakah keluhan pengguna benar-benar terjadi di lokasi bencana.

**BMKG-INTEL** hadir sebagai solusi **Decision Support System (DSS)** yang menggabungkan:
1.  **Kecerdasan Buatan (AI):** Memahami bahasa gaul/slang Indonesia.
2.  **Kecerdasan Lokasi (Geo-Spatial):** Memetakan posisi pelapor.
3.  **Integrasi Data Fisik:** Mengambil data sensor gempa resmi BMKG.

---

## 🚀 Fitur Unggulan

### 🧠 1. AI Engine (IndoBERT Transformer)
Menggunakan model *Pre-trained* **IndoBERT** yang di-*fine-tune* dengan 5.000 data ulasan.
- **Akurasi Tinggi (78.5%):** Jauh lebih unggul dibanding Bi-LSTM atau SVM.
- **Context-Aware:** Mampu membedakan keluhan teknis ("Gagal Login") vs laporan bencana ("Gempa kencang").

### 🗺️ 2. Geo-Spatial Intelligence
- **NER (Named Entity Recognition):** Ekstraksi otomatis nama kota dari teks (misal: "Gempa di *Bandung*").
- **Validasi Silang:** Menampilkan anomali jika ada laporan warga di lokasi yang tidak terdeteksi sensor, atau sebaliknya.
- **Multi-Layer Map:** Peta Jalan (Street) dan Citra Satelit.

### 🛠️ 3. Developer Action Center
- **Bug Radar:** Mengelompokkan isu prioritas (misal: 40 orang lapor notifikasi mati).
- **AI Recommendation:** Memberikan saran perbaikan teknis (coding suggestion) kepada developer.
- **Smart Reply:** Membuat draf balasan otomatis untuk Customer Service.

## 📦 Instalasi & Penggunaan
Ikuti langkah ini untuk menjalankan sistem di komputer lokal Anda.

1. Prasyarat
Python 3.10 atau lebih baru.

Disarankan memiliki GPU NVIDIA (CUDA) untuk performa inferensi AI yang cepat, namun CPU tetap bisa berjalan (agak lambat).

2. Clone Repositori
Bash

git clone [https://github.com/USERNAME/BMKG-Intelligent-Situational-Awareness.git](https://github.com/USERNAME/BMKG-Intelligent-Situational-Awareness.git)
cd BMKG-Intelligent-Situational-Awareness
3. Setup Environment
Bash

### Buat Virtual Environment
python -m venv venv

### Aktifkan (Windows)
venv\Scripts\activate

### Aktifkan (Mac/Linux)
source venv/bin/activate
4. Install Dependencies
Bash

pip install -r requirements.txt
5. Download / Train Model (PENTING!)
Karena file model AI terlalu besar untuk GitHub, Anda harus men-generate modelnya secara lokal pertama kali:

Bash

### Proses ini memakan waktu 30-60 menit tergantung spesifikasi PC
python scripts/01_train_model.py
Script ini akan otomatis mengunduh IndoBERT base dari HuggingFace dan melakukan Fine-Tuning.

6. Jalankan Aplikasi
Bash

python app.py
Buka browser dan akses: http://127.0.0.1:5000

### 👨‍💻 Author
Yudha Rangga WP Magister Teknik Informatika - Universitas Pamulang NIM: 241012000151

### ⚠️ Disclaimer Akademik
Aplikasi ini adalah Purwarupa Penelitian Tesis.

Data Gempa & Cuaca bersumber dari API Terbuka BMKG (Real-time).

Analisis sentimen dihasilkan oleh model AI (IndoBERT) dan mungkin memiliki margin error.

Tidak mewakili pandangan resmi instansi terkait.