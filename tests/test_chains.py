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

    def test_astrapy_filter_selection_chain(self):
        fake_data_access = DataAccess()
        path_segment_keywords = fake_data_access.get_path_segment_keywords()
        model = ChatOpenAI(model_name="gpt-4-1106-preview")
        user_info_summary = {
            "chain1": "Summary for Customer John Smith (Phone Number: 555-555-5555):\n\n1. Billing Query (26-Dec-2023): John contacted customer support regarding a higher-than-usual bill and was informed of additional charges for international calls he made. The support agent, Sarah, provided a one-time discount as a courtesy, and the issue was resolved.\n\n2. Network Issue (26-Dec-2023): John reported poor network coverage at his home. Mike from technical support checked and found ongoing maintenance work likely causing the issue. Resolution is expected within 48 hours and John will be updated via email. The issue status is pending.\n\n3. Device Support (26-Dec-2023): John sought assistance with setting up his new phone, particularly with software configuration, email setup, and data transfer from his old phone. Laura from device support provided step-by-step guidance, and the issue was resolved successfully.\n\nJohn's interactions with support show he has a history of billing awareness, experiences network-related concerns, and requires assistance with device setup. These details may inform future recommendations for service plans with clearer billing details, network updates, or device setup services.",
            "chain2": "Since you haven't provided any specific information to summarize, I can't create a summary for you. Please provide the details or context that you would like to be summarized, and I'll be happy to help.",
            "chain3": "Summary:\n\nThe Smith family consists of nine members with a range of ages from 16 to 50 years old. Their devices vary from older models like the iPhone 6 to newer ones such as the iPhone 13 and OnePlus 9 Pro. Monthly usage across the family averages between 450 and 800 minutes. The family appears to have multiple support cases, with each member having at least one and some having up to three cases associated with their name. All members share the same primary phone number but have individual numbers for family members. The devices used suggest a mix of iOS and Android preferences within the family.\n\nSpecific member details:\n- John Smith, 29, uses an iPhone 6 and averages 680 minutes monthly.\n- Michael Smith, 42, uses a OnePlus 9 Pro with 800 minutes of usage.\n- Emily Smith, 19, has an iPhone SE and uses around 500 minutes.\n- David Smith, 50, prefers a Google Pixel 6 and uses about 700 minutes a month.\n- Olivia Smith, 24, has a Samsung Galaxy S20, with 600 minutes of usage.\n- William Smith, 33, uses an iPhone 13 and averages 750 minutes.\n- Ava Smith, 16, is on a OnePlus Nord and uses 450 minutes.\n- James Smith, 29, uses a Google Pixel 4a with 680 minutes usage.\n- Sophia Smith, 22, has an iPhone XR and uses approximately 520 minutes.\n\nDevice age and usage patterns could influence recommendations for upgrades or tailored plans to better meet the varying needs of the family members.",
        }
        path_segment_keyword_chain = (
            {
                "PathSegmentValues": itemgetter("path_segment_keywords"),
                "UserInformationSummary": itemgetter("user_info_summary"),
            }
            | PromptFactory.build_collection_vector_find_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
        )
        chosen_search_filters = path_segment_keyword_chain.invoke(
            {
                "path_segment_keywords": path_segment_keywords,
                "user_info_summary": user_info_summary,
            }
        )
        print(chosen_search_filters)