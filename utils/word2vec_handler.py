# utils/word2vec_handler.py
from gensim.models import KeyedVectors
import os

class Word2VecHandler:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../models/word2vec.bin')
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def get_similar(self, word, topn=10):
        try:
            results = self.model.most_similar(word, topn=topn)
            return [{"word": w, "score": float(s)} for w, s in results]
        except Exception as e:
            return []
