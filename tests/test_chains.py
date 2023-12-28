from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import unittest
from unittest.mock import MagicMock, patch
from DataAccess import DataAccess
import uuid


class TestChains(unittest.TestCase):
    def test_build_table_mapping_prompt(self, mock_get_session):
        model = ChatOpenAI()
        vectorstore = FAISS.from_texts(
            ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        Answer in the following language: {language}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "language": itemgetter("language"),
            }
            | prompt
            | model
            | StrOutputParser()
        )

        chain.invoke({"question": "where did harrison work", "language": "italian"})
