import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_absa_labeled.csv')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'aspect_model')

# Mapping Label
label2id = {"Akurasi": 0, "UI/UX": 1, "Performa": 2, "Lainnya": 3}
id2label = {0: "Akurasi", 1: "UI/UX", 2: "Performa", 3: "Lainnya"}

class BMKGDataset(torch.utils.data.Dataset):
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
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def main():
    print("="*60)
    print("🚀 MULAI TRAINING INDOBERT (FIXED STRATEGY)")
    print("="*60)

    # 1. Cek GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 2. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")
        return

    df = df[df['Aspek_Terdeteksi'].isin(label2id.keys())]
    texts = df['clean_text'].tolist()
    labels = [label2id[label] for label in df['Aspek_Terdeteksi'].tolist()]

    # Split Data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"📊 Train: {len(train_texts)} | Val: {len(val_texts)}")

    # 3. Tokenizer
    model_checkpoint = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = BMKGDataset(train_encodings, train_labels)
    val_dataset = BMKGDataset(val_encodings, val_labels)

    # 4. Model Init
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # 5. Training Arguments (FIXED HERE)
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, 'models', 'checkpoints'),
        num_train_epochs=3,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        
        # --- PERBAIKAN UTAMA ---
        eval_strategy="steps",    # Evaluasi setiap X steps
        eval_steps=50,            # Cek akurasi tiap 50 steps
        save_strategy="steps",    # Simpan model setiap X steps (HARUS SAMA dengan eval)
        save_steps=50,            # Simpan tiap 50 steps
        save_total_limit=2,       # Hanya simpan 2 model terbaik (hemat storage)
        # -----------------------
        
        load_best_model_at_end=True,
        fp16=True, 
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # 6. Train
    print("\n🏋️‍♂️ Training dimulai...")
    trainer.train()

    # 7. Save Final
    print(f"\n💾 Menyimpan model ke: {MODEL_OUTPUT_DIR}")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    # 8. Test
    print("\n🧪 Tes Prediksi:")
    test_kalimat = "aplikasi ini berat banget bikin hp panas"
    inputs = tokenizer(test_kalimat, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax().item()
    print(f"   Input: '{test_kalimat}'")
    print(f"   Prediksi: {id2label[pred_id]}")

if __name__ == "__main__":
    main()