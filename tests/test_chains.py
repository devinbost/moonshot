import json
from operator import itemgetter
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser, ListOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)

import unittest
from unittest.mock import MagicMock, patch

import ChainFactory
import PromptFactory
from DataAccess import DataAccess
import uuid

from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.PropertyInfo import PropertyInfo
from pydantic_models.TableExecutionInfo import TableExecutionInfo
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo


class TestChains(unittest.TestCase):
    def test_get_chain_to_determine_if_tables_match_user_info_should_not_match(self):
        chain = PromptFactory.get_chain_to_determine_if_tables_match_user_info()
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="Age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="Name",
                    property_type="string",
                    property_value="John Doe",
                ),
            ]
        )
        table_schema = TableSchema(
            table_name="example_table",
            keyspace_name="example_schema",
            columns=[
                ColumnSchema(column_name="id", column_type="int"),
                ColumnSchema(column_name="location", column_type="string"),
            ],
        )
        result = chain.invoke(
            {
                "UserInfo": user_info,
                "TableSchema": table_schema,
            }
        )
        self.assertEqual("NO", result.strip())

    def test_get_chain_to_determine_if_tables_match_user_info_should_match(self):
        chain = PromptFactory.get_chain_to_determine_if_tables_match_user_info()
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="Age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="Name",
                    property_type="string",
                    property_value="John Doe",
                ),
            ]
        )
        table_schema = TableSchema(
            table_name="example_table",
            keyspace_name="example_schema",
            columns=[
                ColumnSchema(column_name="id", column_type="int"),
                ColumnSchema(column_name="name", column_type="string"),
            ],
        )
        result = chain.invoke(
            {
                "UserInfo": user_info,
                "TableSchema": table_schema,
            }
        )
        self.assertEqual("YES", result.strip())

    def test_build_table_mapping_prompt(self, mock_get_session):
        model = ChatOpenAI(model_name="gpt-4-1106-preview")
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

    def test_support_summarization_chain(self):
        fake_data_access = DataAccess()
        model = ChatOpenAI(model_name="gpt-4-1106-preview")

        table_schema = TableSchema(
            keyspace_name="telecom",
            table_name="customer_support_transcripts",
            columns=[
                ColumnSchema(column_name="phone_number", column_type="text"),
                ColumnSchema(column_name="transcript_id", column_type="uuid"),
                ColumnSchema(column_name="customer_name", column_type="text"),
                ColumnSchema(column_name="interaction_date", column_type="timestamp"),
                ColumnSchema(column_name="issue_type", column_type="text"),
                ColumnSchema(column_name="resolution_status", column_type="text"),
                ColumnSchema(column_name="transcript", column_type="text"),
            ],
        )
        fake_data_access.set_table_metadata(table_schema)

        # table_schema.cql_description =
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="name",
                    property_type="text",
                    property_value="John Smith",
                ),
                PropertyInfo(
                    property_name="phone_number",
                    property_type="text",
                    property_value="555-555-5555",
                ),
                PropertyInfo(
                    property_name="email",
                    property_type="text",
                    property_value="johndoe@example.com",
                ),
                PropertyInfo(
                    property_name="address",
                    property_type="text",
                    property_value="123 Main St, Anytown, USA",
                ),
                PropertyInfo(
                    property_name="account_status",
                    property_type="text",
                    property_value="Active",
                ),
                PropertyInfo(
                    property_name="plan_type",
                    property_type="text",
                    property_value="Unlimited Data Plan",
                ),
            ]
        )

        # Need to build TableExecutionInfo from schema for first run
        exec_info = TableExecutionInfo(
            table_schema=table_schema,
            execution_counter=0,
            rows=None,
            prior_failures=None,
        )

        top3_chain = (
            {"TableSchema": itemgetter("table_schema")}
            | PromptFactory.build_select_query_for_top_three_rows()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(fake_data_access.exec_cql_query_simple)
        )
        top3_rows = top3_chain.invoke({"table_schema": table_schema})
        exec_info.rows = top3_rows

        select_with_where_chain = (
            {
                "TableExecutionInfo": itemgetter("TableExecutionInfo"),
                "PropertyInfo": itemgetter("PropertyInfo"),
            }
            | PromptFactory.build_select_query_with_where()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(fake_data_access.exec_cql_query_simple)
        )
        # exec_info = TableExecutionInfo(
        #     table_schema=table_schema, rows=all_where_rows, prior_failures=None
        # )
        table_summarization_chain = (
            {"Information": select_with_where_chain}
            | PromptFactory.build_summarization_prompt()
            | model
            | StrOutputParser()
        )
        summary = table_summarization_chain.invoke(
            {"TableExecutionInfo": exec_info, "PropertyInfo": user_info}
        )
        print(summary)

    def test_user_summarization_chains_simpler(self):
        fake_data_access = DataAccess()
        model = ChatOpenAI(model_name="gpt-4-1106-preview")
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="name",
                    property_type="text",
                    property_value="John Smith",
                ),
                PropertyInfo(
                    property_name="phone_number",
                    property_type="text",
                    property_value="555-555-5555",
                ),
                PropertyInfo(
                    property_name="email",
                    property_type="text",
                    property_value="johndoe@example.com",
                ),
                PropertyInfo(
                    property_name="address",
                    property_type="text",
                    property_value="123 Main St, Anytown, USA",
                ),
                PropertyInfo(
                    property_name="account_status",
                    property_type="text",
                    property_value="Active",
                ),
                PropertyInfo(
                    property_name="plan_type",
                    property_type="text",
                    property_value="Unlimited Data Plan",
                ),
            ]
        )
        family_plans = TableSchema(
            table_name="family_plans",
            keyspace_name="telecom",
            columns=[
                ColumnSchema(
                    column_name="family_member_phone_number",
                    type="text",
                    clustering_order="asc",
                    kind="clustering",
                    position=0,
                ),
                ColumnSchema(
                    column_name="phone_number",
                    type="text",
                    clustering_order="none",
                    kind="partition_key",
                    position=0,
                ),
            ],
            indexes=[],
            keys=None,
        )
        chain = ChainFactory.build_summarization_chain(
            model, fake_data_access, family_plans
        )
        chain_kwargs = {f"chain0": chain}
        # Pass the dictionary as keyword arguments
        summarization_chains = RunnableParallel(**chain_kwargs)
        summary = summarization_chains.invoke(
            {
                "property_info": user_info,
            }
        )
        # Later on, we should incorporate the top k rows of each table to improve matching effectiveness:

        print(summary)

    def test_user_summarization_chains(self):
        fake_data_access = DataAccess()
        model = ChatOpenAI(model_name="gpt-4-1106-preview")
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="name",
                    property_type="text",
                    property_value="John Smith",
                ),
                PropertyInfo(
                    property_name="phone_number",
                    property_type="text",
                    property_value="555-555-5555",
                ),
                PropertyInfo(
                    property_name="email",
                    property_type="text",
                    property_value="johndoe@example.com",
                ),
                PropertyInfo(
                    property_name="address",
                    property_type="text",
                    property_value="123 Main St, Anytown, USA",
                ),
                PropertyInfo(
                    property_name="account_status",
                    property_type="text",
                    property_value="Active",
                ),
                PropertyInfo(
                    property_name="plan_type",
                    property_type="text",
                    property_value="Unlimited Data Plan",
                ),
            ]
        )
        customer_support_transcripts = TableSchema(
            table_name="customer_support_transcripts",
            keyspace_name="telecom",
            columns=[
                ColumnSchema(
                    column_name="phone_number",
                    type="text",
                    clustering_order="none",
                    kind="partition_key",
                    position=0,
                ),
                ColumnSchema(
                    column_name="transcript_id",
                    type="uuid",
                    clustering_order="asc",
                    kind="clustering",
                    position=0,
                ),
            ],
            indexes=[
                "customer_support_transcripts_issue_type_idx",
                "customer_support_transcripts_resolution_status_idx",
            ],
            keys=None,
        )

        family_plan_info = TableSchema(
            table_name="family_plan_info",
            keyspace_name="telecom",
            columns=[
                ColumnSchema(
                    column_name="family_member_phone_number",
                    type="text",
                    clustering_order="asc",
                    kind="clustering",
                    position=0,
                ),
                ColumnSchema(
                    column_name="phone_number",
                    type="text",
                    clustering_order="none",
                    kind="partition_key",
                    position=0,
                ),
            ],
            indexes=[],
            keys=None,
        )

        family_plans = TableSchema(
            table_name="family_plans",
            keyspace_name="telecom",
            columns=[
                ColumnSchema(
                    column_name="family_member_phone_number",
                    type="text",
                    clustering_order="asc",
                    kind="clustering",
                    position=0,
                ),
                ColumnSchema(
                    column_name="phone_number",
                    type="text",
                    clustering_order="none",
                    kind="partition_key",
                    position=0,
                ),
            ],
            indexes=[],
            keys=None,
        )

        tables = [customer_support_transcripts, family_plan_info, family_plans]

        chains = [
            ChainFactory.build_summarization_chain(model, fake_data_access, table)
            for table in tables
        ]
        chain_kwargs = {f"chain{i+1}": chain for i, chain in enumerate(chains)}
        # Pass the dictionary as keyword arguments
        summarization_chains = RunnableParallel(**chain_kwargs)
        summary = summarization_chains.invoke(
            {
                "property_info": user_info,
            }
        )
        # Later on, we should incorporate the top k rows of each table to improve matching effectiveness:

        print(summary)
