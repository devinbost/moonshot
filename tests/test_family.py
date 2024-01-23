import os
import unittest
from unittest.mock import MagicMock, patch
from langchain.chat_models import ChatOpenAI
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, Session
from cassandra.query import SimpleStatement
from astrapy.db import AstraDB as AstraPyDB, AstraDBCollection
from DataAccess import DataAccess
import uuid

from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.TableKey import TableKey
from pydantic_models.TableSchema import TableSchema
from pydantic_models.FamilyData import FamilyData
from pydantic_models.Person import Person
from Config import config
import asyncio
import json
import os
import logging
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from FamilyCrawler import FamilyCrawler
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from pydantic_models.ExecutionFailure import ExecutionFailure
from pydantic_models.TableExecutionInfo import TableExecutionInfo
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo

from langchain_core.runnables import RunnableBranch, RunnableSerializable
from FamilyCrawler import *


class TestData(unittest.TestCase):
    def test_write_to_collection(self):
        # Configure verbose logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        crawler = FamilyCrawler()

        # Initialize the AstraDB for vector storage
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

        # embeddings = OpenAIEmbeddings()
        cassandra_vector_store = AstraDB(
            embedding=embeddings,
            collection_name=os.getenv("ASTRA_COLLECTION"),
            token=os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )

        asyncio.run(
            crawler.process_json_files_async(
                "/Users/devin.bost/proj/repos/moonshot/tests/scratch/json_files",
                cassandra_vector_store,
            )
        )

    def test_family_concatenation(self):
        db = AstraPyDB(
            token=os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        embedding_model = "all-MiniLM-L12-v2"
        embedding_direct = SentenceTransformer(
            "sentence-transformers/" + embedding_model
        )

        query_result = embedding_direct.encode("discovery monument").tolist()
        input_vector = query_result
        collection = AstraDBCollection(collection_name="family_collection", astra_db=db)
        results = collection.vector_find(
            vector=input_vector,
            filter={
                "metadata.recordType": {"$in": ["Mining Records", "Deeds"]},
                "metadata.place": "Iron, Utah, United States",
            },
            limit=6,
        )
        print(results)
        # Need to update Person record to include person's birth_date and death_date

    def test_family_concatenation_nlp(self):
        db = AstraPyDB(
            token=os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        embedding_model = "all-MiniLM-L12-v2"
        embedding_direct = SentenceTransformer(
            "sentence-transformers/" + embedding_model
        )

        query_result = embedding_direct.encode("mortgage").tolist()
        input_vector = query_result
        collection = AstraDBCollection(
            collection_name="family_collection_nlp", astra_db=db
        )
        results = collection.vector_find(
            vector=input_vector,
            filter={
                "metadata.recordType": "Deeds",
                "metadata.names": "Mary Sims",
            },
            limit=6,
        )
        print(results)

    def test_family_concatenation_openai(self):
        db = AstraPyDB(
            token=os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        embedding_model = "all-MiniLM-L12-v2"
        embedding_direct = SentenceTransformer(
            "sentence-transformers/" + embedding_model
        )

        crawler = FamilyCrawler()

        family_data: FamilyData = asyncio.run(
            crawler.extract_family_data(
                f"https://ancestors.familysearch.org/en/KWC3-1DF/moneta-hardy-1903-1980"
            )
        )
        family_summary_with_names: str = "Charles Daley"  # family_data.life_summary

        embeddings = OpenAIEmbeddings()
        query_result = embeddings.embed_query(family_summary_with_names)
        input_vector = query_result
        # input_vector = embedding_direct.encode(family_summary_with_names).tolist()
        collection = AstraDBCollection(
            collection_name="family_collection_openai", astra_db=db
        )
        results = collection.vector_find(
            vector=input_vector,
            filter={"metadata.recordType": "Mining Records"},
            limit=6,
        )
        print(results)

    def test_lcel_family_summarization2(self):
        from langchain_community.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from operator import itemgetter
        from langchain_core.output_parsers import StrOutputParser

        model = ChatOpenAI(model_name="gpt-4-1106-preview")
        prompt = (
            "Create a story from the following family history information. Include any relevant "
            "information you know about what was happening in those areas at those times. Use markdown and make bold "
            "any major life-changing events that might resonate personally with the reader. \n"
            "HISTORIES:"
            "{Histories}"
        )
        template = PromptTemplate.from_template(prompt)
        deep_extended_family_data = asyncio.run(
            extract_deep_extended_family_data(
                "https://ancestors.familysearch.org/en/LWM4-4PM/russell-a.-davies-1899-1983"
            )
        )
        from langchain_core.runnables import RunnableLambda, Runnable

        summaries = consolidate_life_summaries(deep_extended_family_data)
        story_chain = (
            {"Histories": RunnableLambda(lambda x: summaries)}
            | template
            | model
            | StrOutputParser()
        )
        results = story_chain.invoke({})
        print(results)
