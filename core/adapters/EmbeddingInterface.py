class EmbeddingInterface:
    def embed_text(self, text):
        raise NotImplementedError("Subclasses must implement this method")
    def get_model(self):
        raise NotImplementedError("Subclasses must implement this method")