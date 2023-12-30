import logging
from operator import itemgetter
from typing import List

from cassandra.cluster import ResultSet
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from pydantic_models.ExecutionFailure import ExecutionFailure
from pydantic_models.TableExecutionInfo import TableExecutionInfo
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo

from langchain_core.runnables import RunnableBranch, RunnableSerializable


def get_python_code_gen_prefix():
    return f"""You're a helpful assistant. I will give you a scenario, and I want you to generate Python code that performs the task. Don't give me any summary information, explanation, or introduction. Don't say anything other than the code. \n"""


def get_cql_code_gen_prefix():
    return f"""You're a helpful assistant. I will give you a scenario, and I want you to generate valid CQL code (for 
    AstraDB/Cassandra) that performs the task. Don't give me any summary information, explanation, or introduction. 
    Don't say anything other than the code. \n"""


def get_personal_response_prefix():
    return f"""You're a helpful assistant. You're talking to a customer, so be respectful and show empathy. \n"""


def get_helpful_assistant_prefix():
    return f"""You're a helpful assistant. Don't give me any summary information, explanation, or introduction. In 
    fact, don't say anything other than what I specify. \n"""


def build_table_identification_prompt():
    prompt = (
        get_helpful_assistant_prefix()
        + """You are also a great Cassandra database engineer. Your goal is to identify tables that likely contain 
        information that will be relevant for a user (whom I will describe below) so that we can take additional 
        steps to gather more specific information from the relevant tables. I will give you a list of tables, 
        along with their columns and descriptive information, and I will also give you information about that 
        particular user, potentially including conversational history with them, and I want you to return a list with 
        only the tables that have at least one column name that appears to match (or very likely matches) at least 
        one of the properties of the provided user information. I want you to ONLY return the list as a JSON array in 
        the following format: [{{ \"keyspace_name\":\"example_keyspace1\", \"table_name\":\"example_table1\"}}, 
        [{{\"keyspace_name\":\"example_keyspace2\", \"table_name\":\"example_table2\"}}]
        Don't return ANY other information since I need to parse your response as valid JSON.
        
        
        TABLE LIST:
        {TableList}
        
        
        USER INFORMATION:
        {UserInfo}
        
        """
    )
    return PromptTemplate.from_template(prompt)


def build_summarization_prompt() -> PromptTemplate:
    prompt = (
        "You're a helpful assistant. I'm will give you some information, and I want you to summarize what I'm "
        "providing you. This information will be used to either summarize something about a customer or "
        "something we know internally that we will use to make a recommendation, so keep that in mind as you "
        "write the summary. Information that would be obviously irrelevant like UUIDs or numbers without any "
        "context should be omitted, but technical information that might influence a recommendation or decision "
        "should be included if it might be applicable. Don't be wordy, but provide enough detail so that patterns can "
        "be identified when this summary is combined with others. Any device-specific or plan-specific details should "
        'be included.)\n If it appears that no information was provided for you to summarize, just say "N/A"'
        "Here is the information I want you to summarize:"
        ""
        ""
        "{Information}"
    )
    return PromptTemplate.from_template(prompt)


def build_collection_vector_find_prompt() -> PromptTemplate:
    prompt = (
        get_helpful_assistant_prefix()
        + """ Based on the provided summary
        
        Generate a JSON object containing one or more of the following keys: metadata.path_segment_1, 
        metadata.path_segment_2, metadata.path_segment_3, metadata.path_segment_4, metadata.path_segment_5, 
        metadata.path_segment_6 Based on the following VECTOR DATA, I want you to determine which values of those 
        metadata path segment keys are most likely to be associated with the USER INFORMATION below. Each path 
        segment is more specific than the one prior to it. Use the most granular path_segment that makes sense. Avoid 
        using more than one filter but always use at least one. Once you select a filter, I want you to return the 
        information in a list of JSON objects with the following example syntax: [{{"metadata.path_segment_X": "VALUE"}}] where X 
        is the selected path segment value, and VALUE is the value you've chosen based on the PATH SEGMENT VALUES 
        available below. Only select values from the PATH SEGMENT VALUES below. Don't create any other path segment 
        values. Also, ensure that the path segment value you selected corresponds to the correct path_segment number. 
        For example, if the data below shows that 'residential' is associated with path_segment_2, don't use 
        'residential' as a path segment value for any path other than path_segment_2. 
        
        I want you to construct at least 10 (but less than 20) such JSON objects, and they should cover different subjects associated with the provided USER INFORMATION SUMMARY below.
        Try to get good subject coverage. For example, you might select one filter from each of the following categories: plans, phones, support, promos / promotions.
        If you can find matching keywords from the more specific path_segments, please do so.
        
        Return ONLY this list of JSON objects."""
        + """
        PATH SEGMENT VALUES:
        
        {PathSegmentValues}
        
        USER INFORMATION SUMMARY:
        
        {UserInformationSummary}
        """
    )
    return PromptTemplate.from_template(prompt)


def get_relevant_user_tables(tables: list[TableSchema], user_info: UserInfo):
    prefix = get_python_code_gen_prefix()
    prompt = f"""{prefix} \n I will give you a list of table schemas and some user info, and I want you to return the names of tables that have any column name that matches any of the user info property names."""
    for table in tables:
        prompt += table.to_lcel_json()


def build_select_query_for_top_three_rows() -> PromptTemplate:
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n Write a SELECT * query limit 3 for the given table and columns. Return ONLY a SELECT query. NEVER return anything other than a SELECT query.
    {TableSchema}
    
    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_for_top_three_rows_parallelizable(
    table_schema: TableSchema,
) -> PromptTemplate:
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + f"""\n Write a SELECT * query limit 3 for the given table and columns. Return ONLY a SELECT query. NEVER return anything other than a SELECT query.
    {table_schema}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where() -> PromptTemplate:
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some info where at least one of the properties is believed to 
        match a column name. I'm also including some example rows from the table to help you get an idea of what kind 
        of data exists in the table. I want you to construct a SELECT statement where the matching property's value 
        is used in a WHERE clause with its corresponding column name. If a property name closely matches the name of 
        a column, consider that a match. If no such matches are found, if a property's value looks like it is very 
        likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data 
        that matches the user's request. For example, if they ask about a honda, and you have a table named 'cars' 
        with a column named 'make', then you can include WHERE make = 'honda' as a query predicate. However, 
        try not to filter more than necessary. Give me only the complete SELECT statement that you come up with. If 
        execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such 
        failures have occurred, I'll give you the failure reasons we received so you can try to improve the query 
        based on those results to avoid getting the same error. Never use execution_counter or anything from the 
        failure reasons in the actual query parameters you generate. Additionally, I want you to ensure that the 
        SELECT statement you generate is valid CQL based on the available keys and indexes of the table (to the 
        extent possible based on the information provided) and I want you to assume that ALLOW FILTERING is disabled 
        on the table.

    {TableSchema}
    
    {PropertyInfo}
    
    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_parallelizable(
    table_schema: TableSchema,
) -> PromptTemplate:
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + f"""\n I will give you a table schema and some info where at least one of the properties is believed to 
        match a column name. I'm also including some example rows from the table to help you get an idea of what kind 
        of data exists in the table. I want you to construct a SELECT statement where the matching property's value 
        is used in a WHERE clause with its corresponding column name. If a property name closely matches the name of 
        a column, consider that a match. If no such matches are found, if a property's value looks like it is very 
        likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data 
        that matches the user's request. For example, if they ask about a honda, and you have a table named 'cars' 
        with a column named 'make', then you can include WHERE make = 'honda' as a query predicate. However, 
        try not to filter more than necessary. Give me only the complete SELECT statement that you come up with. If 
        execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such 
        failures have occurred, I'll give you the failure reasons we received so you can try to improve the query 
        based on those results to avoid getting the same error. Never use execution_counter or anything from the 
        failure reasons in the actual query parameters you generate. Additionally, I want you to ensure that the 
        SELECT statement you generate is valid CQL based on the available keys and indexes of the table (to the 
        extent possible based on the information provided) and I want you to assume that ALLOW FILTERING is disabled 
        on the table.

    {table_schema}

    {{PropertyInfo}}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_v2() -> PromptTemplate:
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some info where at least one of the properties is believed to 
        match a column name. I'm also including some example rows from the table to help you get an idea of what kind 
        of data exists in the table. I want you to construct a SELECT statement where the matching property's value 
        is used in a WHERE clause with its corresponding column name. If a property name closely matches the name of 
        a column, consider that a match. If no such matches are found, if a property's value looks like it is very 
        likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data 
        that matches the user's request. For example, if they ask about a honda, and you have a table named 'cars' 
        with a column named 'make', then you can include WHERE make = 'honda' as a query predicate. However, 
        try not to filter more than necessary. Give me only the complete SELECT statement that you come up with. If 
        execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such 
        failures have occurred, I'll give you the failure reasons we received so you can try to improve the query 
        based on those results to avoid getting the same error. Never use execution_counter or anything from the 
        failure reasons in the actual query parameters you generate. Additionally, I want you to ensure that the 
        SELECT statement you generate is valid CQL based on the available keys and indexes of the table (to the 
        extent possible based on the information provided) and I want you to assume that ALLOW FILTERING is disabled 
        on the table.

    {TableSchema}
    
    {Top3Rows}

    {PropertyInfo}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_one_variable() -> PromptTemplate:
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some info where at least one of the properties is believed to match a column name. I'm also including
some example rows from the table to help you get an idea of what kind of data exists in the table. 
I want you to construct a SELECT statement where the matching property's value is used in a WHERE clause with its corresponding column name.
If a property name closely matches the name of a column, consider that a match. If no such matches are found, if a property's value looks like it
is very likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data that matches the user's request.
For example, if they ask about a honda, and you have a table named 'cars' with a column named 'make', then you can include WHERE make = 'honda' as 
a query predicate. However, try not to filter more than necessary. Give me only the complete SELECT statement that you come up with.
If execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such failures have occurred, 
I'll give you the failure reasons we received so you can try to improve the query based on those results to avoid getting the same error.
Never use execution_counter or anything from the failure reasons in the actual query parameters you generate.

    {context}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def determine_failure_type() -> PromptTemplate:
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you some information about a failed query, and I want you to categorize the issue type.
        If the error was caused by the LLM token count being exceeded, return "TOKEN_COUNT_EXCEEDED".
        If the error was caused by a syntax error with the query itself, return "SYNTAX_ERROR".
        If the error was caused by something else, return "OTHER".
        Don't return anything else. Only return one of those three values.

    {TableExecutionInfo}

    {TableSchema}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def rewrite_query_based_on_error() -> PromptTemplate:
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you some information about a failed query, and I want you to try writing the query again. 
This time, I want you to fix the error based on the information provided in the error. Be careful to not deviate from the intent
of the original query. NEVER return anything other than a SELECT statement, and no matter what is in the error info, never select
from a table other than the intended one.

    {TableExecutionInfoWithLastFailure}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_where_user_property_matches_column_name() -> PromptTemplate:
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema and some user info, and I want you to determine if any 
    column name matches any of the user info property names. If any such match exists, return a single word response, 
    either "YES" if at least one match is found, or "NO" in any other case.

    {TableSchema}

    {UserInfo}

    RESPONSE:
"""
    )
    template = PromptTemplate.from_template(prompt)
    return template


def trim_to_first_1000_chars(input_str: str) -> str:
    return input_str[:1000]


def clean_string_v2(s: str):
    # Removing backticks and trimming whitespace
    s = s.strip("`' ")

    # Splitting the string into lines
    lines = s.split("\n")

    # Removing any lines that contain only 'sql' or 'json'
    lines = [
        line for line in lines if line.strip().lower() not in ["sql", "json", "cql"]
    ]

    # Joining the lines back into a single string
    return "\n".join(lines)


def get_chain_to_determine_if_table_might_be_relevant_to_user() -> PromptTemplate:
    prefix = get_helpful_assistant_prefix()
    prompt = (
        prefix
        + """\n I will give you a table schema, a few example rows, and some user info, and I want you to determine if any 
    column name matches any of the user info property names. If any such match exists, return a single word response, 
    either "YES" if at least one match is found, or "NO" in any other case.

    {TableSchema}

    {UserInfo}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


# def buildChain(self, table_schema: TableSchema, user_info: UserInfo, data_access: DataAccess):
#     model = ChatOpenAI(model_name="gpt-4-1106-preview")
#
#     # Need to build TableExecutionInfo from schema for first run
#     exec_info = TableExecutionInfo(table_schema=table_schema, rows=None, prior_failures=None)
#
#     top3_chain = (
#         {"TableSchema": itemgetter("table_schema")}
#         | self.build_select_query_for_top_three_rows()
#         | model
#         | StrOutputParser()
#     )
#     top3_query = top3_chain.invoke({"table_schema": table_schema})
#     try:
#         top3_rows = data_access.exec_cql_query("example_namespace", top3_query)
#         exec_info.rows = top3_rows
#     except Exception as ex:
#         logging.error("failure") # Just log the failure for now. Later, we can recursively
#         exec_info.prior_failures.append(ExecutionFailure(top3_query, ex))
#         NotImplementedError()
#
#
#     select_with_where_chain = (
#         {"TableExecutionInfo": itemgetter("exec_info"), "PropertyInfo": itemgetter("user_info")}
#         | self.build_select_query_with_where()
#         | model
#         | StrOutputParser
#     )
#     select_with_where_statement = select_with_where_chain.invoke({"exec_info":exec_info, "user_info": user_info })
#     # Exec the select statement
#     try:
#         all_rows: List[dict] = data_access.exec_cql_query("example_namespace", select_with_where_statement)
#         exec_info.rows = all_rows
#     except Exception as ex:
#         logging.error("failure") # Just log the failure for now. Later, we can recursively
#         exec_info.prior_failures.append(ExecutionFailure(top3_query, ex))
#         NotImplementedError()
#     exec_info = TableExecutionInfo(
#         table_schema=table_schema, rows=all_rows, prior_failures=None
#     )
#     select_with_where_chain2 = (
#         {"TableExecutionInfo": exec_info | increment_counter, "PropertyInfo": user_info}
#         | self.build_select_query_with_where()
#         | model
#         | StrOutputParser
#     )
#
#     # When handling errors, we could have the LLM increment a counter and emit JSON in the response.
#     # Then, we can use JsonKeyOutputFunctionsParser(key_name="mykey") to grab the value from one of the JSON object keys.
#    # func_to_exec_query = ?
#     trim_to_first_1000_chars()
#     smart_select_prompt: PromptTemplate = self.build_select_query_with_where()
#     branch = RunnableBranch(
#         (lambda x: "YES" in x["topic"].lower(), get_chain_to_query_),
#         (lambda x: "NO" in x["topic"].lower(), langchain_chain),
#         determine_if_any_table_columns_match_a_user_property_name(),
#     )
#
#     full_chain.invoke(
#         {
#             "TableSchema": table_schema.to_lcel_json_prefixed(),
#             "UserInfo": user_info.to_lcel_json_prefixed(),
#         }
#     )


def create_synopsis_prompt(title):
    prefix = get_python_code_gen_prefix()
    prompt = PromptTemplate.from_template(
        f"""{prefix}
        I will give you some examples from docs: 

EXAMPLES:

{title}
Playwright: This is a synopsis for the above play:"""
    )
