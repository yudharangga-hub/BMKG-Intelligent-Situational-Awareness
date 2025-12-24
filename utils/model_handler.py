import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import torch.nn.functional as F

class ModelHandler:
    def __init__(self):
        # 1. Deteksi Device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 AI Engine Online: {self.device}")
        
        # 2. Setup Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.absa_path = os.path.join(base_dir, 'models', 'aspect_model')
        self.emotion_path = os.path.join(base_dir, 'models', 'emotion_model')

        # 3. Label Mapping
        self.absa_labels = {0: "Akurasi", 1: "UI/UX", 2: "Performa", 3: "Lainnya"}
        self.emotion_labels = {0: "Marah", 1: "Takut", 2: "Bahagia", 3: "Sedih"}

        # 4. Load Models
        try:
            self.absa_tokenizer = AutoTokenizer.from_pretrained(self.absa_path)
            self.absa_model = AutoModelForSequenceClassification.from_pretrained(self.absa_path).to(self.device).eval()
            
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_path)
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(self.emotion_path).to(self.device).eval()
        except Exception as e:
            print(f"❌ Error Loading Models: {e}")

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()

    def get_model_metadata(self):
        return {
            "absa": {"name": "IndoBERT (Fine-Tuned)", "acc": "78.5%", "arch": "Transformer"},
            "emotion": {"name": "IndoBERT (Fine-Tuned)", "acc": "72.3%", "arch": "Transformer"}
        }

    def generate_recommendations(self, text, aspek, emosi):
        """
        LOGIC CERDAS V3: Context-Aware Recommendation Engine
        Menangani kasus spesifik (Waktu, Widget, GPS) sebelum fallback ke prediksi AI.
        """
        txt = text.lower()
        
        # Default Output
        recs = {
            "action_dev": "Lakukan pengecekan log server pada timestamp laporan.",
            "action_ux": "Evaluasi user journey terkait.",
            "draft_reply": "Terima kasih atas laporannya. Kami akan segera menindaklanjuti."
        }

        # ==========================================
        # KATEGORI 1: ZONA WAKTU (TIMEZONE)
        # ==========================================
        # Kasus: User di Papua/Bali bingung kenapa jam di aplikasi WIB
        if any(x in txt for x in ['waktu', 'jam', 'wib', 'wita', 'wit', 'zona', 'papua', 'bali', 'makassar']):
            if 'salah' in txt or 'beda' in txt or 'atur' in txt or 'bingung' in txt:
                recs["action_dev"] = "🔧 Terapkan `DateTime.now().toLocal()` pada kode aplikasi agar otomatis mengikuti pengaturan jam HP user, bukan jam server Jakarta."
                recs["action_ux"] = "🎨 Tambahkan opsi 'Ganti Zona Waktu' di menu Pengaturan agar user bisa memilih manual (WIB/WITA/WIT)."
                recs["draft_reply"] = "Halo Kak, mohon maaf atas kebingungannya. Saat ini aplikasi memang default menggunakan WIB (Server). Namun, tim kami sedang mengerjakan update agar jam otomatis mengikuti lokasi Kakak (WIT/WITA). Terima kasih masukannya!"
                return recs

        # ==========================================
        # KATEGORI 2: WIDGET & NOTIFIKASI
        # ==========================================
        if 'widget' in txt or 'layar depan' in txt or 'notif' in txt:
            if 'mati' in txt or 'kosong' in txt or 'ilang' in txt or 'muncul' in txt:
                recs["action_dev"] = "🔧 Cek `Background Service` pada Android 12+. Pastikan Widget Service tidak dimatikan oleh fitur 'Battery Saver' bawaan HP."
                recs["action_ux"] = "🎨 Berikan tutorial singkat 'Cara Pasang Widget' saat user pertama kali instal aplikasi."
                recs["draft_reply"] = "Halo Kak, jika widget tidak update/hilang, mohon pastikan fitur 'Penghemat Baterai' tidak membatasi aplikasi BMKG ya. Coba hapus dan pasang ulang widget-nya."
                return recs

        # ==========================================
        # KATEGORI 3: LOKASI & GPS
        # ==========================================
        if 'lokasi' in txt or 'gps' in txt or 'tempat' in txt or 'kota' in txt:
            if 'salah' in txt or 'jauh' in txt or 'ngaco' in txt or 'deteksi' in txt:
                recs["action_dev"] = "🔧 Integrasikan Google Places API untuk akurasi lebih tinggi. Cek izin akses lokasi (Fine Location)."
                recs["action_ux"] = "🎨 Tampilkan nama Kecamatan/Kelurahan di header aplikasi, bukan hanya koordinat angka."
                recs["draft_reply"] = "Halo Kak, pastikan GPS di HP sudah aktif dan izin lokasi diberikan ke aplikasi ya. Terkadang sinyal yang lemah membuat deteksi lokasi meleset ke tower terdekat."
                return recs

        # ==========================================
        # KATEGORI 4: GEMPA BUMI (BENCANA)
        # ==========================================
        if "gempa" in txt or "guncang" in txt or "magnitude" in txt:
            recs["action_dev"] = "🔥 CRITICAL: Pastikan latency Push Notification via FCM di bawah 3 detik."
            recs["action_ux"] = "🎨 Gunakan warna Merah Dominan dan Font Besar saat Mode Warning Gempa aktif."
            recs["draft_reply"] = "Tetap waspada Kak! Kami memprioritaskan kecepatan info gempa. Jika notifikasi telat, kemungkinan karena antrian trafik operator seluler yang padat saat kejadian."
            return recs

        # ==========================================
        # KATEGORI 5: CUACA & HUJAN
        # ==========================================
        if "hujan" in txt or "panas" in txt or "cuaca" in txt or "mendung" in txt:
            recs["action_dev"] = "🔧 Kalibrasi data radar cuaca dengan stasiun pengamatan terdekat."
            recs["action_ux"] = "🎨 Tampilkan persentase 'Peluang Hujan' (misal: 80%) agar user tidak kecewa jika meleset."
            recs["draft_reply"] = "Halo Kak, cuaca tropis sangat dinamis dan bisa berubah hitungan menit. Kami terus mengkalibrasi radar kami agar prediksi semakin akurat. Sedia payung sebelum hujan ya!"
            return recs

        # ==========================================
        # KATEGORI 6: GENERAL UI/UX & PERFORMA (Fallback AI)
        # ==========================================
        # Jika tidak ada kata kunci spesifik di atas, gunakan prediksi Model IndoBERT
        if aspek == "UI/UX":
            recs["action_dev"] = "🔧 Cek responsivitas layout XML pada perangkat dengan DPI rendah/tinggi."
            recs["action_ux"] = "🎨 Lakukan A/B Testing pada menu navigasi. Pertimbangkan Dark Mode jika banyak user mengeluh silau."
            recs["draft_reply"] = "Terima kasih feedback-nya Kak. Kami sadar tampilan perlu penyegaran. Tim desain kami sedang menyiapkan update antarmuka (UI) yang lebih modern."
        
        elif aspek == "Performa":
            recs["action_dev"] = "🔧 Profiling memori (Memory Leak Check). Optimasi query database lokal (SQLite/Realm)."
            recs["action_ux"] = "🎨 Tampilkan 'Skeleton Loading' (bayangan abu-abu) saat data sedang dimuat agar aplikasi tidak terkesan macet."
            recs["draft_reply"] = "Mohon maaf atas kendalanya. Silakan coba 'Clear Cache' atau instal ulang aplikasi. Tim kami terus bekerja keras mengoptimalkan performa server."
        
        elif aspek == "Akurasi":
            recs["action_dev"] = "🔧 Validasi data backend dengan data observasi lapangan."
            recs["action_ux"] = "🎨 Berikan label waktu 'Data Diperbarui: xx menit lalu' agar user tahu validitas data."
            recs["draft_reply"] = "Halo Kak, terima kasih laporannya. Ketepatan data adalah prioritas kami. Laporan ini akan kami jadikan bahan evaluasi tim teknis."

        return recs

    def predict(self, text):
        clean_txt = self.clean_text(text)
        
        # 1. Prediksi AI (Deep Learning)
        inp_absa = self.absa_tokenizer(clean_txt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        inp_emo = self.emotion_tokenizer(clean_txt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            prob_absa = F.softmax(self.absa_model(**inp_absa).logits, dim=1)
            conf_absa, pred_absa = torch.max(prob_absa, dim=1)
            
            prob_emo = F.softmax(self.emotion_model(**inp_emo).logits, dim=1)
            conf_emo, pred_emo = torch.max(prob_emo, dim=1)

        aspek = self.absa_labels[pred_absa.item()]
        emosi = self.emotion_labels[pred_emo.item()].title()
        
        # 2. Generate Logic (Rule-Based Expert System)
        recommendations = self.generate_recommendations(text, aspek, emosi)

        return {
            "aspek": aspek,
            "aspek_conf": round(conf_absa.item() * 100, 1),
            "emosi": emosi,
            "emosi_conf": round(conf_emo.item() * 100, 1),
            "recommendations": recommendations 
        }