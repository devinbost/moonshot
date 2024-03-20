from core.adapters.EmbeddingInterface import EmbeddingInterface


class AzureOpenAIEmbeddingAdapter(EmbeddingInterface):
    def __init__(self, model):
        self.model = model

    def embed_text(self, text):
        query_result = self.model.embed_query(text)
        # Assuming `embed_query` returns the embedding in the desired format; adjust as needed
        return query_result

    def get_model(self):
        return self.model