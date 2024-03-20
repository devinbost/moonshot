from core.adapters.EmbeddingInterface import EmbeddingInterface


class SentenceTransformerAdapter(EmbeddingInterface):
    def __init__(self, model):
        self.model = model

    def embed_text(self, text):
        return self.model.encode(text).tolist()

    def get_model(self):
        return self.model