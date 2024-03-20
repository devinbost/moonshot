from core.adapters.EmbeddingInterface import EmbeddingInterface


class HuggingFaceEmbeddingAdapter(EmbeddingInterface):
    def __init__(self, model):
        self.model = model

    def embed_text(self, text):
        return self.model.embed_query(text)

    def get_model(self):
        return self.model