from operator import itemgetter

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import PromptFactory
from DataAccess import DataAccess
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo


def build_summarization_chain(
    model: ChatOpenAI, data_access: DataAccess, table_schema: TableSchema
):
    top3_chain = (
        PromptFactory.build_select_query_for_top_three_rows_parallelizable(table_schema)
        | model
        | StrOutputParser()
        | RunnableLambda(PromptFactory.clean_string_v2)
        | RunnableLambda(data_access.exec_cql_query_simple)
    )
    select_with_where_chain = (
        {
            "Top3Rows": top3_chain,
            "PropertyInfo": itemgetter("property_info"),
        }
        | PromptFactory.build_select_query_with_where_parallelizable(table_schema)
        | model
        | StrOutputParser()
        | RunnableLambda(PromptFactory.clean_string_v2)
        | RunnableLambda(data_access.exec_cql_query_simple)
    )
    table_summarization_chain = (
        {"Information": select_with_where_chain}
        | PromptFactory.build_summarization_prompt()
        | model
        | StrOutputParser()
    )
    return table_summarization_chain


def build_collection_summarization_chain(
    model: ChatOpenAI, data_access: DataAccess, table_schema: TableSchema
):
    top3_chain = (
        PromptFactory.build_select_query_for_top_three_rows_parallelizable(table_schema)
        | model
        | StrOutputParser()
        | RunnableLambda(PromptFactory.clean_string_v2)
        | RunnableLambda(data_access.exec_cql_query_simple)
    )
    select_with_where_chain = (
        {
            "Top3Rows": top3_chain,
            "PropertyInfo": itemgetter("property_info"),
        }
        | PromptFactory.build_select_query_with_where_parallelizable(table_schema)
        | model
        | StrOutputParser()
        | RunnableLambda(PromptFactory.clean_string_v2)
        | RunnableLambda(data_access.exec_cql_query_simple)
    )
    table_summarization_chain = (
        {"Information": select_with_where_chain}
        | PromptFactory.build_summarization_prompt()
        | model
        | StrOutputParser()
    )
    return table_summarization_chain
