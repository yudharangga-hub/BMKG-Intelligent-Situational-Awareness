import os
import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_absa_labeled.csv')
EMO_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_emotion_labeled.csv')
OUTPUT_JSON = os.path.join(BASE_DIR, 'static', 'model_metrics.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path, data_path, text_col, label_col, label_map):
    print(f"📊 Evaluating model: {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    try:
        # LOAD FULL DATA (Tanpa Sample 200)
        df = pd.read_csv(data_path)
        
        # Normalisasi Label (Title Case)
        df[label_col] = df[label_col].astype(str).str.strip().str.title()
        
        # Filter Label Valid
        valid_labels = list(label_map.values())
        df = df[df[label_col].isin(valid_labels)]
        
        # --- PENTING UNTUK AKADEMIK ---
        # Kita gunakan 20% data sebagai Test Set (Data yang tidak dilihat saat training)
        # Agar validasi objektif (bukan testing on training data)
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Jika ingin melihat SEMUA data (tanpa split), uncomment baris bawah ini & comment baris atas:
        # test_df = df 

        texts = test_df[text_col].astype(str).tolist()
        true_labels = test_df[label_col].tolist()
        
        if len(texts) == 0: return None
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    preds = []
    print(f"   Testing on {len(texts)} real samples (20% Split)...")
    
    # Batch Processing agar lebih cepat
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        pred_ids = torch.argmax(logits, dim=1).tolist()
        for pid in pred_ids:
            preds.append(label_map[pid])

    # Hitung Metrics
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    
    unique_labels = list(label_map.values())
    cm = confusion_matrix(true_labels, preds, labels=unique_labels)
    
    return {
        "accuracy": round(acc * 100, 2),
        "f1": round(f1, 2),
        "cm": cm.tolist(),
        "labels": unique_labels
    }

def main():
    print("🚀 GENERATING FULL METRICS (HIGH VOLUME)...")
    
    # 1. Evaluasi ABSA
    absa_path = os.path.join(BASE_DIR, 'models', 'aspect_model')
    absa_map = {0: "Akurasi", 1: "UI/UX", 2: "Performa", 3: "Lainnya"}
    absa_metrics = evaluate_model(absa_path, DATA_PATH, 'clean_text', 'Aspek_Terdeteksi', absa_map)

    # 2. Evaluasi Emotion
    emo_path = os.path.join(BASE_DIR, 'models', 'emotion_model')
    emo_map = {0: "Marah", 1: "Takut", 2: "Bahagia", 3: "Sedih"}
    emo_metrics = evaluate_model(emo_path, EMO_PATH, 'clean_text', 'Emosi', emo_map)

    # 3. Simpan
    if absa_metrics and emo_metrics:
        final_data = {"absa": absa_metrics, "emotion": emo_metrics}
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(final_data, f, indent=4)
        print(f"\n✅ SUCCESS! Full Metrics saved to: {OUTPUT_JSON}")
    else:
        print("\n❌ FAILED.")

if __name__ == "__main__":
    main()