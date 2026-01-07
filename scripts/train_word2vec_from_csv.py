# Script: train_word2vec_from_csv.py
# Train Word2Vec model from data/raw/arsip_scraping_lengkap.csv
# Output: models/word2vec/word2vec.model

import pandas as pd
import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Path setup
data_path = os.path.join('data', 'raw', 'arsip_scraping_lengkap.csv')
output_dir = os.path.join('models', 'word2vec')
os.makedirs(output_dir, exist_ok=True)
output_model = os.path.join(output_dir, 'word2vec.model')

# Load data
print('Loading CSV...')
df = pd.read_csv(data_path)

# Preprocess: ambil kolom komentar, tokenisasi
print('Preprocessing...')
sentences = [simple_preprocess(str(text)) for text in df['Komentar'].dropna()]

# Train Word2Vec
print('Training Word2Vec...')
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=1)

# Save model (Gensim .model)
temp_path = output_model + '.tmp'
model.save(temp_path)
os.replace(temp_path, output_model)
print(f'Model saved to {output_model}')

# Save model in word2vec .bin format for handler compatibility
bin_path = os.path.join(output_dir, 'word2vec.bin')
model.wv.save_word2vec_format(bin_path, binary=True)
print(f'Model (bin) saved to {bin_path}')
