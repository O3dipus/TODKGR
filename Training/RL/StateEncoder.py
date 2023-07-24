from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api


class StateEncoder:
    def __init__(self):
        self.glove_model = api.load("glove-wiki-gigaword-100")

    def get_encoded(self, word):
        embedding = self.glove_model[word]
        return embedding
