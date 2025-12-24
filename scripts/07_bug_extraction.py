import pandas as pd
import os
import json
from collections import Counter

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_absa_labeled.csv')
OUTPUT_JSON = os.path.join(BASE_DIR, 'static', 'bug_report.json')

def categorize_issue(text):
    """
    Mengelompokkan teks keluhan ke dalam kategori masalah spesifik.
    Ini membuat laporan lebih mudah dibaca manusia daripada sekadar n-gram.
    """
    text = str(text).lower()
    
    # 1. Kategori Performa / Teknis
    if any(x in text for x in ['lemot', 'lambat', 'berat', 'lag', 'macet', 'stuck']):
        return "Aplikasi Lambat / Berat"
    if any(x in text for x in ['keluar sendiri', 'force close', 'fc', 'crash', 'tutup']):
        return "Force Close / Crash"
    if any(x in text for x in ['gagal login', 'masuk', 'daftar', 'otp']):
        return "Masalah Login / Akun"
    if any(x in text for x in ['koneksi', 'jaringan', 'internet', 'server', 'down']):
        return "Koneksi / Server Down"
    
    # 2. Kategori Fitur Gempa
    if 'gempa' in text:
        if any(x in text for x in ['notif', 'bunyi', 'suara', 'alarm', 'telat']):
            return "Notifikasi Gempa Terlambat/Mati"
        if any(x in text for x in ['lokasi', 'titik', 'peta', 'koordinat']):
            return "Akurasi Lokasi Gempa"
        return "Info Gempa Tidak Update"

    # 3. Kategori Cuaca
    if any(x in text for x in ['cuaca', 'hujan', 'panas', 'mendung']):
        if any(x in text for x in ['salah', 'beda', 'ngaco', 'tidak sesuai']):
            return "Prediksi Cuaca Tidak Akurat"
        if any(x in text for x in ['widget', 'tampilan']):
            return "Widget Cuaca Bermasalah"
    
    # 4. Kategori UI/UX
    if any(x in text for x in ['iklan', 'banyak iklan']):
        return "Terlalu Banyak Iklan"
    if any(x in text for x in ['update', 'versi baru']):
        return "Bug Setelah Update Aplikasi"
    if any(x in text for x in ['gelap', 'mode malam', 'tulisan', 'huruf']):
        return "Masalah Tampilan / UI"

    return None # Tidak masuk kategori utama

def get_recommendation(issue):
    """Memberikan saran teknis berdasarkan kategori masalah"""
    recs = {
        "Aplikasi Lambat / Berat": "Lakukan profiling memori & optimasi query database lokal.",
        "Force Close / Crash": "Cek log 'Fatal Exception' pada Android Vitals & perbaiki NullPointer.",
        "Masalah Login / Akun": "Periksa API Gateway & layanan OTP provider.",
        "Koneksi / Server Down": "Scale-up kapasitas server saat traffic tinggi & cek CDN.",
        "Notifikasi Gempa Terlambat/Mati": "Prioritaskan push notification channel 'High Importance' di Firebase.",
        "Akurasi Lokasi Gempa": "Validasi koordinat sensor seismograf dengan peta digital.",
        "Info Gempa Tidak Update": "Pastikan sinkronisasi data background berjalan real-time.",
        "Prediksi Cuaca Tidak Akurat": "Kalibrasi model prediksi dengan data stasiun pengamatan terdekat.",
        "Widget Cuaca Bermasalah": "Perbaiki service widget agar auto-refresh di background.",
        "Terlalu Banyak Iklan": "Kurangi frekuensi iklan interstitial agar tidak mengganggu UX.",
        "Bug Setelah Update Aplikasi": "Rollback fitur bermasalah atau rilis hotfix secepatnya.",
        "Masalah Tampilan / UI": "Evaluasi kontras warna & ukuran font untuk aksesibilitas."
    }
    return recs.get(issue, "Lakukan investigasi log lebih lanjut.")

def main():
    print("🐛 SMART BUG DETECTION RUNNING...")
    
    if not os.path.exists(DATA_PATH):
        print("❌ Data not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Filter hanya sentimen negatif
    neg_df = df[df['Sentimen'] == 'Negatif']
    
    issues_found = []
    
    # Scan setiap komentar
    for text in neg_df['Komentar']:
        issue = categorize_issue(text)
        if issue:
            issues_found.append(issue)
            
    # Hitung frekuensi masalah
    issue_counts = Counter(issues_found).most_common(10) # Top 10 Masalah
    
    # Format JSON Output
    report = {
        "critical": [],
        "ux_issues": []
    }
    
    for issue, count in issue_counts:
        item = {
            "issue": issue,
            "count": count,
            "recommendation": get_recommendation(issue)
        }
        
        # Pisahkan mana Critical (Teknis) vs UX (Tampilan)
        if any(x in issue for x in ['Lambat', 'Crash', 'Login', 'Server', 'Notifikasi', 'Akurasi']):
            report['critical'].append(item)
        else:
            report['ux_issues'].append(item)

    # Simpan
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"✅ Smart Bug Report Generated: {OUTPUT_JSON}")
    # Preview
    for item in report['critical'][:3]:
        print(f"   - {item['issue']}: {item['count']}")

if __name__ == "__main__":
    main()