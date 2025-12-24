import os
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import sys

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'arsip_scraping_lengkap.csv')
DATA_LABELED = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_emotion_labeled.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'emotion_model')

# Definisi Label Emosi
emotion_labels = ["marah", "takut", "bahagia", "sedih"]
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for i, label in enumerate(emotion_labels)}

# --- FUNGSI UTILITIES ---
def clean_text(text):
    if not isinstance(text, str): return ""
    import re
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def main():
    print("="*60)
    print("🎭 MEMULAI PIPELINE EMOTION DETECTION (AUTO-LABEL + TRAIN)")
    print("="*60)

    # 1. Setup Device
    device = 0 if torch.cuda.is_available() else -1
    print(f"✅ GPU: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 2. Cek apakah sudah ada data berlabel?
    if os.path.exists(DATA_LABELED):
        print("📂 Data berlabel emosi ditemukan! Skip proses auto-labeling.")
        df = pd.read_csv(DATA_LABELED)
    else:
        print("⚠️ Data emosi belum ada. Memulai AUTO-LABELING dengan Zero-Shot...")
        df = pd.read_csv(DATA_RAW)
        df['clean_text'] = df['Komentar'].apply(clean_text)
        df = df[df['clean_text'].str.strip() != '']
        
        # Load Zero-Shot Model
        classifier = pipeline("zero-shot-classification", 
                              model="joeddav/xlm-roberta-large-xnli", 
                              device=device)
        
        preds = []
        batch_size = 32
        texts = df['clean_text'].tolist()
        
        print("🚀 Sedang melabeli emosi (Marah/Takut/Bahagia/Sedih)...")
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            if not batch: continue
            try:
                results = classifier(batch, emotion_labels, multi_label=False)
                for res in results:
                    preds.append(res['labels'][0])
            except Exception as e:
                for _ in batch: preds.append("netral") # Fallback
        
        df['Emosi'] = preds
        df.to_csv(DATA_LABELED, index=False)
        print(f"✅ Data berlabel tersimpan: {DATA_LABELED}")

    # 3. Persiapan Training IndoBERT
    print("\n🏋️‍♂️ Memulai Training Model Emosi (IndoBERT)...")
    texts = df['clean_text'].tolist()
    # Filter hanya label valid
    valid_idxs = [i for i, label in enumerate(df['Emosi']) if label in label2id]
    texts = [texts[i] for i in valid_idxs]
    labels = [label2id[df['Emosi'].iloc[i]] for i in valid_idxs]

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    print(f"📊 Train Size: {len(train_texts)} | Val Size: {len(val_texts)}")

    model_checkpoint = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, 'models', 'emotion_checkpoints'),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True, 
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    trainer.train()

    # 4. Simpan Model
    print(f"\n💾 Menyimpan Model Emosi ke: {MODEL_DIR}")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # 5. Test
    print("\n🧪 Tes Prediksi Emosi:")
    test_cases = ["Gempa kencang sekali saya takut!", "Terima kasih infonya sangat membantu", "Admin lambat update bikin emosi"]
    inputs = tokenizer(test_cases, padding=True, truncation=True, return_tensors="pt").to(device if torch.cuda.is_available() else "cpu")
    model.to(device if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    preds = torch.argmax(logits, dim=1)
    for text, pred in zip(test_cases, preds):
        print(f"   📝 '{text}' -> {id2label[pred.item()].upper()}")

if __name__ == "__main__":
    main()