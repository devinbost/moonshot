from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L12-v2"):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.embedding_direct = SentenceTransformer(
            f"sentence-transformers/{model_name}"
        )

    def get_sentence_transformer(self):
        return self.embedding_direct
