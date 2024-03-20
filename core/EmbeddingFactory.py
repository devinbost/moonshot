from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
from langchain_openai import AzureOpenAIEmbeddings

from core.adapters.AzureOpenAIEmbeddingAdapter import AzureOpenAIEmbeddingAdapter
from core.adapters.SentenceTransformerAdapter import SentenceTransformerAdapter


class EmbeddingFactory:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def create_embedding(self, embedding_type, **kwargs):
        if embedding_type == "sentence_transformer":
            model_name = self.config_loader.get("embedding.sentence_transformer.required.MODEL_NAME")
            model_name = kwargs.get("model_name", model_name)
            model = SentenceTransformer(f"sentence-transformers/{model_name}")
            return SentenceTransformerAdapter(model)

        elif embedding_type == "azure":
            azure_deployment = self.config_loader.get("embedding.azure.required.AZURE_DEPLOYMENT")
            openai_api_version = self.config_loader.get("embedding.azure.required.OPENAI_API_VERSION")
            model = AzureOpenAIEmbeddings(
                azure_deployment=azure_deployment,
                openai_api_version=openai_api_version
            )
            return AzureOpenAIEmbeddingAdapter(model)

        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

