import json
import logging
from operator import itemgetter
from typing import List, Dict, Any

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, Runnable

import PromptFactory
from DataAccess import DataAccess
from core.CollectionManager import CollectionManager
from core.LLMFactory import LLMFactory
from pydantic_models.QAPair import QAPair
from pydantic_models.TableSchema import TableSchema


class ChainFactory:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory

    def build_summarization_chain(
        self, model: ChatOpenAI, data_access: DataAccess, table_schema: TableSchema
    ) -> Runnable:
        """
        Builds a chain for summarizing table data using a specified model and data access object.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            data_access (DataAccess): The data access object for executing queries.
            table_schema (TableSchema): The schema of the table to be summarized.
        Returns:
            Runnable: A runnable chain for table data summarization.
        """

        def test_func(testme: Dict[str, Any]) -> Any:
            print(testme)
            print(table_schema)
            user_props = testme["user_info_not_summary"]
            return user_props

        select_with_where_chain: Runnable = (
            {
                # "Top3Rows": top3_chain,
                "PropertyInfo": test_func,  # Should be user properties of some kind
            }
            | PromptFactory.build_select_query_with_where_parallelizable(table_schema)
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(data_access.exec_cql_query_simple_async)
        )
        table_summarization_chain: Runnable = (
            {"Information": select_with_where_chain}
            | PromptFactory.build_summarization_prompt()
            | self.llm_factory.create_llm_model("openai", model_name="gpt-3.5-turbo-1106")
            | StrOutputParser()
        )
        return table_summarization_chain

    def build_keyword_reduction_prompt_chain(
        self, model: ChatOpenAI, user_info_summary: Any, keywords: List[str]
    ):
        def clean(x):
            try:
                y = json.loads(x)
                return y
            except Exception as ex:
                temp = "{" + x + "}"
                y = json.loads(temp)
                return y

        path_segment_keyword_chain = (
            {
                "Keywords": RunnableLambda(lambda x: keywords),
                "UserInformationSummary": RunnableLambda(lambda x: user_info_summary),
            }
            | PromptFactory.build_keyword_reduction_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(lambda x: clean(x))
        )
        return path_segment_keyword_chain

    def build_vector_search_summarization_chain(
        self, model: ChatOpenAI, search_results: str, llm_character_cut_off: int = 16000
    ) -> Runnable:
        """
        Builds a chain for summarizing vector search results.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            search_results (str): The search results to be summarized.
            llm_character_cut_off (int): The character limit for the LLM - after this value, everything is trimmed to prevent the LLM from giving an error due to too many tokens
        Returns:
            Runnable: A runnable chain for vector search result summarization.
        """
        if search_results is not None:
            collection_summary_chain = (
                {"Information": RunnableLambda(lambda x: search_results[:llm_character_cut_off])}
                | PromptFactory.build_summarization_prompt()
                | model
                | StrOutputParser()
            )
            return collection_summary_chain

    def build_final_recommendation_chain_non_parallel(
        self,
        model: ChatOpenAI,
    ):
        final_chain = (
            {
                "UserSummary": itemgetter("user_summary"),
                "BusinessSummary": itemgetter("business_summary"),
                "UserMessages": itemgetter("user_messages"),
                "OurResponses": itemgetter("our_responses")
                | RunnableLambda(lambda x: x.reverse()),
            }
            | PromptFactory.build_final_response_prompt()
            | model.bind(stop="Best regards,")
            | StrOutputParser()
        )
        return final_chain

    async def get_relevant_tables(self, data_access, model, user_info):
        relevant_table_chain = (
            {
                "TableList": RunnableLambda(data_access.get_table_schemas_in_db_v2),
                "UserInfo": itemgetter("user_info_not_summary"),
            }
            | PromptFactory.build_table_identification_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(data_access.map_tables_and_populate_async)
        )
        relevant_tables = await relevant_table_chain.ainvoke(
            {"user_info_not_summary": user_info}
        )
        return relevant_tables

    def build_question_generation_chain_from_summary(
            self, model: ChatOpenAI
    ) -> Runnable:
        """Output json is ['Question1', 'Question2', ..., 'QuestionN']"""
        chain = (
                {
                    "CustomerSummary": itemgetter("table_summary"),
                    "UserMessage": itemgetter("user_message"),
                    "QuestionCount": itemgetter("question_count")
                }
                | PromptFactory.build_questions_from_user_summary()
                | model
                | StrOutputParser()
                | RunnableLambda(PromptFactory.clean_string_v2)
                | RunnableLambda(lambda x: json.loads(x))
        )
        return chain

    async def invoke_answer_chain(
        self, qa_chain, collection_manager: CollectionManager, question: str, vector_search_limit: int
    ) -> QAPair:
        search_results = await collection_manager.ANN_search_async(question, vector_search_limit)
        answer = await qa_chain.ainvoke({"search_results": search_results,
                                   "question": question})

        qa_pair = QAPair(question = question, answer = answer)
        return qa_pair

    def build_answer_chain(self, model: ChatOpenAI) -> Runnable:
        """Output is the answer to the provided question"""

        chain = (
                {
                    "SearchResults": itemgetter("search_results"),
                    "Question": itemgetter("question")
                }
                | PromptFactory.rag_qa_chain()
                | model
                | StrOutputParser()
        )
        return chain
