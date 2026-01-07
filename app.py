
"""
Urutan kode sudah dirapikan:
1. Semua import
2. Inisialisasi Flask, AI, BMKGHandler, Word2Vec
3. Semua route Flask
"""
# =============================
# IMPORT & INISIALISASI FLASK
# =============================
from flask import Flask, render_template, request, jsonify
from utils.model_handler import ModelHandler
from utils.bmkg_api import BMKGHandler
from utils.word2vec_handler import Word2VecHandler
import pandas as pd
import os
import threading
import re

app = Flask(__name__)

# ==========================================
# 1. SYSTEM BOOT: LOAD RESOURCES
# ==========================================
print("🔌 SYSTEM BOOT: Initializing Neural Networks & Data Streams...")

# A. Load AI Model
try:
    ai_brain = ModelHandler()
    print("✅ AI CORE: ONLINE (IndoBERT Loaded)")
except Exception as e:
    print(f"❌ AI CORE ERROR: {e}")
    ai_brain = None

# B. Load BMKG API Handler
try:
    bmkg_feed = BMKGHandler()
    print("✅ BMKG FEED: READY")
except Exception as e:
    print(f"⚠️ BMKG FEED ERROR: {e}")
    bmkg_feed = None

# C. Load Word2Vec
word2vec_model = None
def load_word2vec():
    global word2vec_model
    try:
        word2vec_model = Word2VecHandler()
        print('✅ Word2Vec Model Loaded')
    except Exception as e:
        print(f'❌ Word2Vec Load Error: {e}')

threading.Thread(target=load_word2vec).start()

# ==========================================
# 5. API CHATBOT (INFORMASI GEMPA & CUACA)
# ==========================================
@app.route('/api/chatbot', methods=['POST'])
def api_chatbot():
    """Chatbot: Jawab pertanyaan cuaca, gempa, warning"""
    data = request.json
    msg = data.get('message', '').lower()
    if not msg:
        return jsonify({"reply": "Mohon masukkan pertanyaan."})

    # Intent: Gempa (lebih toleran)
    if any(k in msg for k in ["gempa", "guncang", "magnitude", "dimana gempa", "lokasi gempa", "terkini"]):
        try:
            print(f"[Chatbot] Query gempa: {msg}")
            if not bmkg_feed:
                print("[Chatbot] bmkg_feed tidak tersedia!")
                return jsonify({"reply": "Data gempa tidak tersedia."})
            quake = bmkg_feed.get_latest_quake()
            if not quake:
                print("[Chatbot] Data gempa kosong!")
                return jsonify({"reply": "Tidak ada data gempa terbaru."})
            reply = f"Gempa terbaru: Magnitudo {quake['magnitudo']} SR di {quake['wilayah']} pada {quake['jam']}. Kedalaman {quake['kedalaman']}. Potensi: {quake['potensi']}."
            print(f"[Chatbot] Jawaban: {reply}")
            return jsonify({"reply": reply})
        except Exception as e:
            print(f"[Chatbot] ERROR gempa: {e}")
            return jsonify({"reply": "Maaf, terjadi error saat mengambil data gempa."})

    # Intent: Cuaca (deteksi kota)
    if "cuaca" in msg or "hujan" in msg or "panas" in msg:
        try:
            print(f"[Chatbot] Query cuaca: {msg}")
            kota = None
            for city in bmkg_feed.cities:
                # Cek apakah nama kota (tanpa spasi, lowercase) ada di pertanyaan
                if city['name'].lower().replace(' ', '') in msg.replace(' ', ''):
                    kota = city['name']
                    break
            if not kota:
                # Coba fallback: cari substring kota (tanpa spasi)
                for city in bmkg_feed.cities:
                    if city['name'].split()[0].lower() in msg:
                        kota = city['name']
                        break
            if not kota:
                print("[Chatbot] Kota tidak ditemukan di query.")
                return jsonify({"reply": "Sebutkan nama kota untuk info cuaca. Contoh: 'Cuaca di Jakarta?'"})
            print(f"[Chatbot] Kota terdeteksi: {kota}")
            weather_list = bmkg_feed.get_all_weather()
            print(f"[Chatbot] Jumlah data cuaca: {len(weather_list)}")
            for w in weather_list:
                if kota.lower() in w['kota'].lower():
                    reply = f"Cuaca di {w['kota']}, {w['provinsi']}: {w['desc']}, Suhu {w['suhu']}°C, Humiditas {w['humid']}%, Angin {w['angin']} km/jam."
                    print(f"[Chatbot] Jawaban: {reply}")
                    return jsonify({"reply": reply})
            print(f"[Chatbot] Data cuaca untuk {kota} tidak ditemukan di hasil API.")
            return jsonify({"reply": f"Data cuaca untuk {kota} tidak ditemukan."})
        except Exception as e:
            print(f"[Chatbot] ERROR cuaca: {e}")
            return jsonify({"reply": "Maaf, terjadi error saat mengambil data cuaca."})

    # Intent: Warning
    if "peringatan" in msg or "warning" in msg:
        try:
            warnings = bmkg_feed.get_weather_warning() if bmkg_feed else []
            if not warnings:
                return jsonify({"reply": "Tidak ada peringatan cuaca/gempa saat ini."})
            reply = "\n\n".join([f"{w['judul']}: {w['deskripsi']}" for w in warnings])
            return jsonify({"reply": reply})
        except Exception as e:
            return jsonify({"reply": "Maaf, terjadi error saat mengambil data peringatan BMKG."})

    # Default fallback
    return jsonify({"reply": "Maaf, saya hanya bisa menjawab pertanyaan tentang gempa, cuaca, atau peringatan BMKG."})


# ==========================================
# 2. DATA MEMORY: CSV STATISTICS
# ==========================================
DATA_METRICS = {
    "total": "0", 
    "size": "0 KB", 
    "last_update": "-",
    "status": "Offline"
}

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'raw', 'arsip_scraping_lengkap.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        file_size_kb = os.path.getsize(csv_path) / 1024
        
        last_date_str = "-"
        if 'Tanggal' in df.columns:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            if not df['Tanggal'].isnull().all():
                last_date_str = df['Tanggal'].max().strftime('%d %b %Y')

        DATA_METRICS = {
            "total": f"{total_rows:,}".replace(",", "."),
            "size": f"{file_size_kb:.1f} KB",
            "last_update": last_date_str,
            "status": "Active"
        }
        print(f"✅ DATA MEMORY: LOADED ({total_rows} rows)")
    else:
        print("⚠️ DATA MEMORY: CSV Not Found")

except Exception as e:
    print(f"❌ DATA STAT ERROR: {e}")


# ==========================================
# 3. WEB ROUTES (PAGES)
# ==========================================

@app.route('/')
def home():
    """Halaman Utama (Investigator & Dashboard)"""
    return render_template('index.html', metrics=DATA_METRICS)

@app.route('/map')
def map_page():
    """Halaman Geo-Spatial Map"""
    return render_template('ner_map.html')

@app.route('/trends')
def trends_page():
    """Halaman Time Series Analysis"""
    return render_template('trends.html')


@app.route('/dev_recommendations')
def dev_page():
    """Halaman Developer Action Center"""
    return render_template('dev_dashboard.html')

# Semantic Lab Route
@app.route('/semantic_lab')
def semantic_lab():
    """Halaman Semantic Lab (Word2Vec Slang Thesaurus)"""
    return render_template('semantic_lab.html')


# ==========================================
# 4. API ENDPOINTS (JSON DATA)
# ==========================================

# Endpoint Word2Vec untuk Semantic Lab
@app.route('/api/word2vec')
def api_word2vec():
    word = request.args.get('word', '').strip().lower()
    if not word:
        return jsonify({"error": "Parameter 'word' kosong."}), 400
    if not word2vec_model or not hasattr(word2vec_model, 'get_similar'):
        return jsonify({"error": "Word2Vec model belum siap."}), 500
    results = word2vec_model.get_similar(word)
    if not results:
        return jsonify({"error": f"Tidak ditemukan sinonim/asosiasi untuk '{word}'."}), 404
    return jsonify({"word": word, "similar": results})

@app.route('/analyze', methods=['POST'])
def analyze():
    """API Analisis Sentimen (AI)"""
    if not ai_brain: return jsonify({"error": "AI System Not Loaded"}), 500
    
    data = request.json
    text = data.get('text', '')
    if not text: return jsonify({"error": "Input text empty"}), 400
    
    try:
        result = ai_brain.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """API Metadata Model"""
    if not ai_brain: return jsonify({})
    return jsonify(ai_brain.get_model_metadata())

@app.route('/api/live_quake')
def api_live_quake():
    """Proxy API Gempa BMKG (Latest & Recent)"""
    if not bmkg_feed: return jsonify({"error": "BMKG Handler Error"}), 500
    
    latest = bmkg_feed.get_latest_quake()
    recent = bmkg_feed.get_recent_quakes()
    return jsonify({"latest": latest, "recent": recent})

@app.route('/api/live_weather')
def api_live_weather():
    """Proxy API Cuaca Multi-Kota"""
    if not bmkg_feed: return jsonify([])
    weather_data = bmkg_feed.get_all_weather()
    return jsonify(weather_data)

@app.route('/api/weather_warning')
def api_weather_warning():
    """Proxy API Peringatan Dini (CAP)"""
    if not bmkg_feed: return jsonify([])
    warnings = bmkg_feed.get_weather_warning()
    return jsonify(warnings)


if __name__ == '__main__':
    print("\n🚀 SERVER READY! Access at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)