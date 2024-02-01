import json
from operator import itemgetter
from typing import List, Dict, Any

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, Runnable

import PromptFactory
from DataAccess import DataAccess
from pydantic_models.TableSchema import TableSchema


class ChainFactory:
    def __init__(self):
        self.model35: ChatOpenAI = ChatOpenAI(model_name="gpt-3.5-turbo-1106")

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
            | self.model35
            | StrOutputParser()
        )
        return table_summarization_chain

    @DeprecationWarning
    def build_summarization_chain_set(
        self, model: ChatOpenAI, data_access: DataAccess, tables: List[TableSchema]
    ) -> RunnableParallel:
        """
        Builds a set of summarization chains for a list of table schemas.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            data_access (DataAccess): The data access object for executing queries.
            tables (List[TableSchema]): A list of table schemas to be summarized.
        Returns:
            RunnableParallel: A parallel runnable chain for summarizing multiple tables.
        """
        chains = [
            self.build_summarization_chain(model, data_access, table)
            for table in tables
        ]
        chain_kwargs = {f"chain{i+1}": chain for i, chain in enumerate(chains)}
        # Pass the dictionary as keyword arguments
        summarization_chains = RunnableParallel(**chain_kwargs)
        return summarization_chains

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

    def build_collection_predicate_chain(
        self,
        model: ChatOpenAI,
        table_summarization: str,
        all_keywords: Dict[str, List[str]],
    ) -> Runnable:
        """
        Builds a non-parallel version 2 chain for collection predicate extraction.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            table_summarization (str): The summarization of table data.
            all_keywords (Dict[str, List[str]]): A dictionary of all keywords to be considered.
        Returns:
            Runnable: A runnable chain for collection predicate extraction.
            that Returns list of filters like this: [{{"metadata.path_segment_X": "VALUE"}}]
        """
        all_keywords_string: str = json.dumps(all_keywords)
        # print("All Keywords String:", all_keywords_string)
        # print("Table Summarization:", table_summarization)

        path_segment_values_lambda = RunnableLambda(lambda x: all_keywords_string)
        user_information_summary_lambda = RunnableLambda(lambda x: table_summarization)

        # print("Path Segment Values Lambda:", path_segment_values_lambda)
        # print("User Information Summary Lambda:", user_information_summary_lambda)
        #
        # prompt_factory_result = PromptFactory.build_collection_vector_find_prompt_v2()
        # print("Prompt Factory Result:", prompt_factory_result)
        path_segment_keyword_chain = (
            {
                "PathSegmentValues": path_segment_values_lambda,
                "UserInformationSummary": user_information_summary_lambda,
            }
            | PromptFactory.build_collection_vector_find_prompt_v4()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(lambda x: json.loads(x))
        )
        return path_segment_keyword_chain

    def build_vector_search_summarization_chain(
        self, model: ChatOpenAI, search_results: str
    ) -> Runnable:
        """
        Builds a chain for summarizing vector search results.
        Parameters:
            model (ChatOpenAI): The model to be used for generating prompts and processing responses.
            search_results (str): The search results to be summarized.
        Returns:
            Runnable: A runnable chain for vector search result summarization.
        """
        if search_results is not None:
            collection_summary_chain = (
                {"Information": RunnableLambda(lambda x: search_results[:16000])}
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
