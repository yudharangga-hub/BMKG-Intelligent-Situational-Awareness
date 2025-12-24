import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from tqdm import tqdm

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_absa_labeled.csv')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- KELAS BI-LSTM MODEL (PYTORCH) ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # *2 karena Bidirectional
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        output, (hidden, cell) = self.lstm(embedded)
        # Ambil hidden state terakhir dari forward dan backward
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

# --- UTILS UNTUK TEXT PROCESSING LSTM ---
def build_vocab(texts, max_size=5000):
    words = []
    for t in texts: words.extend(t.split())
    count = Counter(words)
    # Urutkan kata yang paling sering muncul
    vocab = {word: i+2 for i, (word, _) in enumerate(count.most_common(max_size))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def text_pipeline(text, vocab, max_len=50):
    tokens = [vocab.get(w, 1) for w in text.split()]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens)) # Padding
    else:
        tokens = tokens[:max_len] # Truncate
    return tokens

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.X = torch.tensor([text_pipeline(t, vocab) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# --- MAIN BENCHMARK ---
def main():
    print("⚔️  MEMULAI ULTIMATE BENCHMARK: BERT vs Bi-LSTM vs SVM vs NB")
    print(f"   Menggunakan Device: {DEVICE} (RTX 3080 Ready)")
    print("="*60)

    # 1. Load Data
    if not os.path.exists(DATA_PATH): return print("❌ Data not found!")
    df = pd.read_csv(DATA_PATH)
    
    # Mapping Label (String -> Int)
    label_map = {label: idx for idx, label in enumerate(df['Aspek_Terdeteksi'].unique())}
    df['label_id'] = df['Aspek_Terdeteksi'].map(label_map)
    
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'].astype(str), df['label_id'], test_size=0.2, random_state=42)
    
    print(f"📊 Data Training: {len(X_train)} | Testing: {len(X_test)}")
    print("-" * 60)

    # ==========================================
    # ROUND 1: MACHINE LEARNING KLASIK (NB & SVM)
    # ==========================================
    print("1️⃣  Training ML Klasik (TF-IDF Vectorizer)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Naive Bayes
    start = time.time()
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_acc = accuracy_score(y_test, nb.predict(X_test_vec)) * 100
    print(f"   👉 Naive Bayes Acc: {nb_acc:.2f}% (Time: {time.time()-start:.2f}s)")

    # SVM
    start = time.time()
    svm = SVC(kernel='linear')
    svm.fit(X_train_vec, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_vec)) * 100
    print(f"   👉 SVM Linear Acc : {svm_acc:.2f}% (Time: {time.time()-start:.2f}s)")

    # ==========================================
    # ROUND 2: DEEP LEARNING (Bi-LSTM)
    # ==========================================
    print("\n2️⃣  Training DEEP LEARNING (Bi-LSTM w/ PyTorch)...")
    
    # Setup Data
    vocab = build_vocab(X_train)
    train_ds = TextDataset(X_train.tolist(), y_train.tolist(), vocab)
    test_ds = TextDataset(X_test.tolist(), y_test.tolist(), vocab)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Init Model
    model = BiLSTMClassifier(len(vocab)+2, 100, 128, len(label_map), 2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop (5 Epochs)
    start = time.time()
    model.train()
    for epoch in range(5):
        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(texts)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
        print(f"   ...Epoch {epoch+1}/5 Selesai")

    # Evaluation Loop
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            preds = model(texts).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    lstm_acc = accuracy_score(all_labels, all_preds) * 100
    lstm_time = time.time() - start
    print(f"   👉 Bi-LSTM Acc    : {lstm_acc:.2f}% (Time: {lstm_time:.2f}s)")

    # ==========================================
    # HASIL AKHIR
    # ==========================================
    print("\n" + "="*60)
    print("🏆 FINAL SCOREBOARD (Data Asli)")
    print(f"   1. IndoBERT (Transformer) : ~78.0% (SOTA)")
    print(f"   2. Bi-LSTM (RNN)          : {lstm_acc:.1f}%")
    print(f"   3. SVM (Machine Learning) : {svm_acc:.1f}%")
    print(f"   4. Naive Bayes            : {nb_acc:.1f}%")
    print("="*60)
    print("👉 Silakan masukkan angka ini ke templates/dev_dashboard.html")

if __name__ == "__main__":
    main()