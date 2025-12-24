from flask import Flask, render_template, request, jsonify
from utils.model_handler import ModelHandler
from utils.bmkg_api import BMKGHandler
import pandas as pd
import os

# Inisialisasi Flask
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


# ==========================================
# 4. API ENDPOINTS (JSON DATA)
# ==========================================

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