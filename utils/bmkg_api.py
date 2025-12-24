import requests
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

class BMKGHandler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.url_gempa_latest = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
        self.url_gempa_list = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"
        self.url_warning_rss = "https://www.bmkg.go.id/alerts/nowcast/id/rss.xml"

        # --- DAFTAR KOTA REPRESENTATIF SELURUH INDONESIA (MAJOR CITIES) ---
        # Kode adm4 diambil sampel dari wilayah ibukota provinsi/kota besar
        self.cities = [
            # SUMATERA
            {"name": "Banda Aceh", "code": "11.71.02.1001"},
            {"name": "Medan", "code": "12.71.02.1001"},
            {"name": "Padang", "code": "13.71.02.1001"},
            {"name": "Pekanbaru", "code": "14.71.02.1001"},
            {"name": "Palembang", "code": "16.71.02.1001"},
            {"name": "Bengkulu", "code": "17.71.02.1001"},
            {"name": "Bandar Lampung", "code": "18.71.02.1001"},
            
            # JAWA
            {"name": "Jakarta Pusat", "code": "31.71.01.1002"},
            {"name": "Bandung", "code": "32.73.02.1001"},
            {"name": "Semarang", "code": "33.74.02.1001"},
            {"name": "Yogyakarta", "code": "34.71.02.1001"},
            {"name": "Surabaya", "code": "35.78.02.1001"},
            {"name": "Serang", "code": "36.73.02.1001"},

            # BALI & NUSA TENGGARA
            {"name": "Denpasar", "code": "51.71.01.1001"},
            {"name": "Mataram", "code": "52.71.01.1001"},
            {"name": "Kupang", "code": "53.71.01.1001"},

            # KALIMANTAN
            {"name": "Pontianak", "code": "61.71.01.1001"},
            {"name": "Palangkaraya", "code": "62.71.01.1001"},
            {"name": "Banjarmasin", "code": "63.71.01.1001"},
            {"name": "Samarinda", "code": "64.72.01.1001"},
            {"name": "IKN (Sepaku)", "code": "64.09.04.2001"}, # Ibu Kota Nusantara

            # SULAWESI
            {"name": "Manado", "code": "71.71.01.1001"},
            {"name": "Palu", "code": "72.71.01.1001"},
            {"name": "Makassar", "code": "73.71.11.1001"},
            {"name": "Kendari", "code": "74.71.01.1001"},
            {"name": "Gorontalo", "code": "75.71.01.1001"},
            {"name": "Mamuju", "code": "76.04.03.1001"},

            # MALUKU & PAPUA
            {"name": "Ambon", "code": "81.71.01.1001"},
            {"name": "Ternate", "code": "82.71.01.1001"},
            {"name": "Jayapura", "code": "91.71.01.1001"},
            {"name": "Manokwari", "code": "92.02.12.1001"},
            {"name": "Sorong", "code": "92.71.01.1001"},
            {"name": "Merauke", "code": "93.01.01.1001"}
        ]

    # --- 1. GEMPA BUMI ---
    def get_latest_quake(self):
        """Ambil 1 Gempa Terkini + Shakemap Image"""
        try:
            r = requests.get(self.url_gempa_latest, headers=self.headers, timeout=10)
            if r.status_code == 200:
                g = r.json()['Infogempa']['gempa']
                return {
                    "magnitudo": g['Magnitude'],
                    "kedalaman": g['Kedalaman'],
                    "koordinat": g['Coordinates'],
                    "wilayah": g['Wilayah'],
                    "jam": f"{g['Tanggal']} - {g['Jam']}",
                    "potensi": g['Potensi'],
                    "dirasakan": g.get('Dirasakan', '-'),
                    "shakemap": "https://data.bmkg.go.id/DataMKG/TEWS/" + g['Shakemap']
                }
        except Exception as e:
            print(f"Error Latest Quake: {e}")
        return None

    def get_recent_quakes(self):
        """Ambil 15 Gempa Terkini"""
        try:
            r = requests.get(self.url_gempa_list, headers=self.headers, timeout=10)
            if r.status_code == 200:
                return [{
                    "magnitudo": g['Magnitude'], "kedalaman": g['Kedalaman'],
                    "wilayah": g['Wilayah'], "koordinat": g['Coordinates'],
                    "jam": f"{g['Tanggal']} - {g['Jam']}", "potensi": g['Potensi']
                } for g in r.json()['Infogempa']['gempa']]
        except: return []

    # --- 2. CUACA (MULTI KOTA) ---
    def fetch_single_weather(self, city):
        """Helper Cuaca Per Kota"""
        try:
            url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={city['code']}"
            r = requests.get(url, headers=self.headers, timeout=5)
            if r.status_code == 200:
                d = r.json()['data'][0]['cuaca'][0][0]
                loc = r.json()['lokasi']
                return {
                    "kota": city['name'], 
                    "provinsi": loc['provinsi'],
                    "lat": loc['lat'], 
                    "lon": loc['lon'],
                    "desc": d['weather_desc'], 
                    "suhu": d['t'],
                    "humid": d['hu'], 
                    "angin": d['ws'], 
                    "angin_dir": d['wd'],
                    "icon": d['image']
                }
        except Exception as e:
            # print(f"⚠️ Weather Error ({city['name']}): {e}") # Silent error agar console bersih
            return None

    def get_all_weather(self):
        """Ambil Cuaca Multi-Kota (Parallel Processing)"""
        # Gunakan max_workers=10 agar pengambilan 30+ kota lebih cepat
        with ThreadPoolExecutor(max_workers=10) as ex:
            return [d for d in list(ex.map(self.fetch_single_weather, self.cities)) if d]

    # --- 3. WARNING (PERINGATAN DINI) ---
    def get_weather_warning(self):
        """Ambil Peringatan Dini Cuaca (RSS XML)"""
        warnings = []
        try:
            r = requests.get(self.url_warning_rss, headers=self.headers, timeout=10)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                channel = root.find('channel')
                for item in channel.findall('item')[:5]:
                    warnings.append({
                        "judul": item.find('title').text,
                        "link": item.find('link').text,
                        "waktu": item.find('pubDate').text,
                        "deskripsi": item.find('description').text
                    })
        except Exception as e:
            print(f"⚠️ Warning Feed Error: {e}")
        return warnings