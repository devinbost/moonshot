from cassandra.cluster import (
    Cluster,
)
from cassandra.auth import PlainTextAuthProvider

from astrapy.db import AstraDB as AstraPyDB
from astrapy.db import AsyncAstraDB as AsyncAstraPyDB
from langchain.vectorstores import Cassandra
from langchain.vectorstores import AstraDB

from core.ConfigLoader import ConfigLoader
from core.EmbeddingFactory import EmbeddingFactory
from core.adapters.EmbeddingInterface import EmbeddingInterface


class VectorStoreFactory:
    def __init__(
        self, embedding: EmbeddingInterface, config_loader: ConfigLoader
    ):
        self.embedding = embedding
        self.config_loader = config_loader

    def create_vector_store(self, store_type, **kwargs):
        # if store_type == "Cassandra":  # Native API interface for LangChain
        #     return Cassandra(
        #         embedding=self.embedding_manager.get_embedding(),
        #         session=kwargs.get("session"),
        #         keyspace=self.config_loader.get("keyspace"),
        #         table_name=self.config_loader.get("table_name"),
        #     )
        # elif
        if store_type == "AstraDB":  # Data API interface for LangChain
            # Future version should use this instead: https://github.com/langchain-ai/langchain-datastax/tree/main/libs/astradb
            return AstraDB(
                embedding=self.embedding.get_model(),
                collection_name=kwargs.get("collection_name"),
                token=self.config_loader.get("astradb.ASTRA_TOKEN"),
                api_endpoint=self.config_loader.get("astradb.ASTRA_ENDPOINT"),
            )
        elif store_type == "AstraPyDB":  # Data API interface for direct access
            return AsyncAstraPyDB(
                token=self.config_loader.get("astradb.ASTRA_TOKEN"),
                api_endpoint=self.config_loader.get("astradb.ASTRA_ENDPOINT"),
            )
        elif store_type == "Cluster":  # Native API interface for CQL
            return Cluster(
                cloud={
                    "secure_connect_bundle": self.config_loader.get(
                        "astradb.SECURE_BUNDLE_PATH"
                    ),
                },
                auth_provider=PlainTextAuthProvider(
                    "token",
                    self.config_loader.get("astradb.ASTRA_TOKEN"),
                ),
            )
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
