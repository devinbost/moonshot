from cassandra.cluster import (
    Cluster,
    Session,
)
from cassandra.auth import PlainTextAuthProvider
import os
import json

from graphviz import Digraph
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Cassandra
import wget
import pandas as pd
from langchain.docstore.document import Document

import config
from ComponentData import ComponentData


class DataAccess:
    def __init__(self):
        self.vector_store = None
        self.keyspace = os.getenv("KEYSPACE", "keyspace")
        self.table_name = os.getenv("TABLE_NAME", "table")
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="all-mpnet-base-v2"
        # )
        self.output_variables = ["new", "myllm", "emb"]
        self.data_map = {}
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=2000,
            length_function=len,
            is_separator_regex=False,
        )
        cloud_config = {
            "secure_connect_bundle": config.scratch_path + "/secure-connect-openai.zip"
        }
        print(cloud_config)
        with open(config.openai_json) as f:
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
        sample_data = config.data_path + "/20220301.simple.csv"

        if not os.path.isfile(sample_data):
            wget.download(url)
        data = pd.read_csv(sample_data)
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

    def get_output_variable_names(self):
        return self.output_variables

    def add_output_variable(self, variable):
        self.output_variables.append(variable)

    def add_component(self, component_data: ComponentData):
        self.data_map[component_data.id] = component_data
        if component_data.output_var is not None:
            self.add_output_variable(component_data.output_var)

    def get_data_map(self):
        return self.data_map

    def build_graph(self, component_dict: dict[str, ComponentData], graph: Digraph):
        # Style for nodes
        graph.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="lightgrey",
            fontname="Helvetica",
        )

        # Add nodes with additional attributes
        for left_id, left_data in component_dict.items():
            label = f"{left_data.component_name} | Class: {left_data.class_name} | Library: {left_data.library} | Access: {left_data.access_type} | Params: {left_data.params} | Output: {left_data.output_var}"
            graph.node(left_id, label=label)
            print(f"Adding graph.node({left_id}, label={label})")

        # Add edges
        for left_id, left_data in component_dict.items():
            for right_id, right_data in component_dict.items():
                if left_id == right_id:
                    continue
                if left_data.output_var in right_data.params.values():
                    left_label = f"{left_data.component_name} | Class: {left_data.class_name} | Library: {left_data.library} | Access: {left_data.access_type} | Params: {left_data.params}"
                    right_label = f"{right_data.component_name} | Class: {right_data.class_name} | Library: {right_data.library} | Access: {right_data.access_type} | Params: {right_data.params}"
                    print(f"left_label is {left_label})")
                    print(f"right_label is {right_label})")
                    graph.edge(left_id, right_id, label=left_data.output_var)
                    print(
                        f"Adding graph.edge({left_id}, {right_id}, label={left_data.output_var})"
                    )
        return graph
