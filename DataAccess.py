from cassandra.cluster import (
    Cluster,
    Session,
)
from cassandra.auth import PlainTextAuthProvider
import os
import json

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Cassandra
import wget
import pandas as pd
from langchain.docstore.document import Document


class DataAccess:
    def __init__(self):
        self.vector_store = None
        self.keyspace = os.getenv("KEYSPACE")
        self.table_name = os.getenv("TABLE_NAME")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=125,
            length_function=len,
            is_separator_regex=False,
        )
        cloud_config = {"secure_connect_bundle": "secure-connect-openai.zip"}
        with open("devin.bost@datastax.com-token.json") as f:
            secrets = json.load(f)

        self.token = secrets[
            "token"
        ]  # This isn't really safe for production, but it's okay for a demo.
        self.secure_bundle_path = (
            os.getcwd() + "/" + cloud_config["secure_connect_bundle"]
        )
        self.vector_store = self._setupVectorStore()

    def getCqlSession(self) -> Session:
        cluster = Cluster(
            cloud={
                "secure_connect_bundle": self.secure_bundle_path,
            },
            auth_provider=PlainTextAuthProvider(
                "token",
                os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"),
            ),
        )

        astra_session = cluster.connect()
        return astra_session

    def _setupVectorStore(self) -> Cassandra:
        return Cassandra(
            embedding=self.embeddings,
            session=self.getCqlSession(),
            keyspace=self.keyspace,
            table_name=self.table_name,
        )

    def getVectorStore(self) -> Cassandra:
        return self.vector_store

    def loadWikipediaData(self):
        url = "https://raw.githubusercontent.com/GeorgeCrossIV/Langchain-Retrieval-Augmentation-with-CASSIO/main/20220301.simple.csv"
        if not os.path.isfile("./20220301.simple.csv"):
            wget.download(url)
        data = pd.read_csv("20220301.simple.csv")
        data = data.head(10)
        data = data.rename(columns={"text ": "text"})
        for index, row in data.iterrows():
            self._parseWikipediaRow(row)

    def _parseWikipediaRow(self, row):
        metadata = {"url": row["url"], "title": row["title"]}
        page_content = row["text"]

        wikiDocument = Document(page_content=page_content, metadata=metadata)
        wikiDocs = self.splitter.transform_documents([wikiDocument])
        self.vector_store.add_documents(wikiDocs)
