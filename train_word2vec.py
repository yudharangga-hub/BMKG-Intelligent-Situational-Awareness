import pandas as pd
from gensim.models import Word2Vec
import re

# 1. Load data (ganti path sesuai data Anda)
df = pd.read_csv('data/raw/arsip_scraping_lengkap.csv')
texts = df['Komentar'].astype(str).tolist()

# 2. Preprocessing sederhana (tokenisasi)
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
    return text.split()

sentences = [simple_tokenize(t) for t in texts]

# 3. Training Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=1)

# 4. Simpan model ke format binary
model.wv.save_word2vec_format('models/word2vec.bin', binary=True)
print('✅ Model Word2Vec berhasil disimpan ke models/word2vec.bin')