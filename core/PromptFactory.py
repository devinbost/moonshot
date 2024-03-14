from langchain.prompts import PromptTemplate
from pydantic_models.TableSchema import TableSchema


def get_python_code_gen_prefix() -> str:
    """
    Returns a prefix for generating Python code generation prompts.
    This prefix instructs the assistant to focus solely on generating Python code based on a given scenario without any additional explanation or summary.
    Returns:
        str: The prefix string for Python code generation prompts.
    """
    return f"""You're a helpful assistant. I will give you a scenario, and I want you to generate Python code that performs the task. Don't give me any summary information, explanation, or introduction. Don't say anything other than the code. \n"""


def get_cql_code_gen_prefix() -> str:
    """
    Returns a prefix for generating CQL (Cassandra Query Language) code generation prompts.
    This prefix guides the assistant to generate valid CQL code for AstraDB/Cassandra tasks without offering any extra information or context.
    Returns:
        str: The prefix string for CQL code generation prompts.
    """
    return f"""You're a helpful assistant. I will give you a scenario, and I want you to generate valid CQL code (for 
    AstraDB/Cassandra) that performs the task. Don't give me any summary information, explanation, or introduction. 
    Don't say anything other than the code. \n"""


def get_personal_response_prefix() -> str:
    """
    Returns a prefix for generating personal and empathetic responses in customer interactions.
    This prefix sets the tone for respectful and empathetic communication with customers.
    Returns:
        str: The prefix string for personal response generation.
    """
    return f"""You're a helpful assistant. You're talking to a customer, so be respectful and show empathy.
    \n"""


def get_helpful_assistant_prefix() -> str:
    """
    Returns a prefix for generating concise and direct responses from the assistant.
    This prefix instructs the assistant to provide responses strictly adhering to the specified request without any additional information or elaboration.
    Returns:
        str: The prefix string for concise and direct assistant responses.
    """
    return f"""You're a helpful assistant. Don't give me any summary information, explanation, or introduction. In 
    fact, don't say anything other than what I specify. \n"""


def build_table_identification_prompt() -> PromptTemplate:
    """
    Builds and returns a prompt template for identifying relevant tables based on user information and table schemas.
    The prompt aims to select tables with columns that match or are likely to match user properties, helping to focus on relevant data.
    Returns:
        PromptTemplate: The constructed prompt template for table identification.
    """
    prompt: str = (
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


def build_final_response_prompt() -> PromptTemplate:
    """
    Constructs a prompt template for generating a final recommendation to a customer.
    The prompt synthesizes user summaries, business insights, and conversation history to formulate a recommendation that aligns with the customer's needs and interests.
    Returns:
        PromptTemplate: The constructed prompt template for final response generation.
    """
    prompt = (
        get_personal_response_prefix()
        + """"You are responsible for representing the business to the customer for both sales and customer support 
purposes. Your ability to address the user's intent is critical to the success of our business. You will be 
given a rich set of summaries of basically everything we know about this customer, and then you will be given 
a second set of summaries with a lot of information that is likely relevant to their needs based on a 
previously run search algorithm that we ran internally on their information. Your job is to use this 
information to make the best possible recommendation to the customer. The recommendation should be grounded 
in what we know about them and our business, based on the information we obtained. Recommend what is best for 
the customer, but if it's good for the business also, that's a double win. You will also be provided (at the 
end) with the customer's most recent messages, ordered from oldest to most recent. Be sure that your 
recommendation to them is relevant to their messages, especially their most recent message. Always do what is 
in the customer's best interest. You are responsible for providing all of the relevant information to the 
customer. The information in the BUSINESS SUMMARIES section is from the company. The information in the USER 
SUMMARIES section is from the customer. Provide any details like pricing or terms that might help them make a 
decision. Don't say anything that we already said to the customer but always make a recommendation, 
especially if it will help them avoid bill shock. Make just one really solid recommendation, rather than 
several smaller ones. Focus on making the one best recommendation that you think is the most compelling and 
will help the customer the most. However, if they ask a question, be sure to answer the question based on their information. In that case, it's okay to reply with two paragraphs, the first paragraph answering their question and the second paragraph giving them a recommendation. Don't provide a phone number. Conclude with "Best regards,".
Try to be concise, but explain the reasons behind why your recommendation is well suited to them based on what you know about their personal circumstance.
Keep the response to 4 sentences or less. 
Output the response in well formatted markdown that uses bullet points, bold, or any other structure that might help the response to be easily readable. 
Include main points in a bulleted list. 
If the customer asks a question, don't give your recommendation until you've sufficiently answered their question.
Also, make sure the headings are appropriate for the customer. For example, instead of saying, "Answer to Customer's Question:", say "Answer to Your Question".
Summaries:

==================

USER SUMMARIES:
{UserSummary}

==================

BUSINESS SUMMARIES:
{BusinessSummary}

==================

USER SUMMARIES:
{UserSummary}

==================

USER MESSAGES:
{UserMessages}

==================

OUR PRIOR RESPONSES:
{OurResponses}
        """
    )
    return PromptTemplate.from_template(prompt)


def build_summarization_prompt() -> PromptTemplate:
    """
    Creates a prompt template for summarizing provided information, focusing on technical details that could influence recommendations or decisions.
    The prompt is intended to yield concise summaries that highlight key information relevant to customer support or business insights.
    Returns:
        PromptTemplate: The constructed prompt template for summarization.
    """
    prompt = (
        "You're a helpful assistant. I'm will give you some information, and I want you to give a very detailed summary of what I'm "
        "providing you. This information will be used to either summarize something about a customer or "
        "something we know internally that we will use to make a recommendation. "
        "You want to focus your summary around technical information that might influence a "
        "recommendation or decision. If there are relevant prices or numbers, be sure to include them."
        "Provide enough detail so that patterns can "
        "be identified when this summary is combined with others. Any device-specific, plan-specific details, or pricing details should "
        "be included.)\n Be sure to summarize every topic that you're given. (Don't skip anything.) Ignore anything that's repeated more than twice. If the information I provide is all blank after I say "
        '"Here is the information I want you to summarize:", return "articles not actually relevant"'
        "Here is the information I want you to summarize:"
        ""
        ""
        "{Information}"
    )
    return PromptTemplate.from_template(prompt)


def build_keyword_reduction_prompt() -> PromptTemplate:
    prompt = (
        get_helpful_assistant_prefix()
        + """
I will give you a list of keywords (like 5g-mobile-gaming) within a KEYWORDS section, and I will give you some user information within a USER INFORMATION section.
I want you to select only those keywords from the KEYWORDS section that match information in the USER INFORMATION section.
Return the top 3 best matches. ONLY return keywords that exist in the KEYWORDS section. DO NOT return any keyword from the USER INFORMATION section
unless it also exists in the KEYWORDS section. Keywords may or may not contain hyphens, but they are always delimited by a comma or new line character.
Return only the filtered list of keywords as a single JSON array in the format:
["key-word1", "key-word2", . . . , "keywordN"]

===========
KEYWORDS:
{Keywords}

===========
USER INFORMATION SUMMARY:
{UserInformationSummary}
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_for_top_three_rows() -> PromptTemplate:
    """
    Builds a prompt template for generating a SELECT query to retrieve the top three rows from a given table schema.
    The prompt specifically requests a SELECT query with a LIMIT of 3, adhering to CQL syntax.
    Returns:
        PromptTemplate: The constructed prompt template for SELECT query generation.
    """
    prefix = get_cql_code_gen_prefix()
    prompt = (
        prefix
        + """\n Write a SELECT * query limit 3 for the given table and columns. Return ONLY a SELECT query. NEVER return anything other than a SELECT query.
    {TableSchema}
    
    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def get_select_query_with_where_text() -> str:
    prefix = get_cql_code_gen_prefix()
    result = prefix + (
        "\n I will give you a table schema and some info where at least one of the properties is believed to "
        "match a column name. I'm also including some example rows from the table to help you get an idea of what kind "
        "of data exists in the table. I want you to construct a SELECT statement where the matching property's value "
        "is used in a WHERE clause with its corresponding column name. If a property name closely matches the name of "
        "a column, consider that a match. If no such matches are found, if a property's value looks like it is very "
        "likely to be useful in a where clause, you can use it, but remember that the goal is to find relevant data "
        "that matches the user's request. For example, if they ask about a honda, and you have a table named 'cars' "
        "with a column named 'make', then you can include WHERE make = 'honda' as a query predicate. However, "
        "try not to filter more than necessary. Give me only the complete SELECT statement that you come up with. If "
        "execution_counter is greater than 0, then it means the previous attempt to write a query failed, and if such "
        "failures have occurred, I'll give you the failure reasons we received so you can try to improve the query "
        "based on those results to avoid getting the same error. Never use execution_counter or anything from the "
        "failure reasons in the actual query parameters you generate. Additionally, I want you to ensure that the "
        "SELECT statement you generate is valid CQL based on the available keys and indexes of the table (to the "
        "extent possible based on the information provided) and I want you to assume that ALLOW FILTERING is disabled "
        "on the table."
    )
    return result


def build_select_query_with_where() -> PromptTemplate:
    """
    Creates a prompt template for formulating a SELECT query with a WHERE clause based on matching table schema properties and user information.
    The prompt aims to generate a relevant SELECT query that filters data according to user-related criteria, ensuring query validity and relevance.
    Returns:
        PromptTemplate: The constructed prompt template for SELECT query generation with a WHERE clause.
    """
    prompt = (
        get_select_query_with_where_text()
        + """
    {TableSchema}
    
    {PropertyInfo}
    
    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_parallelizable(
    table_schema: TableSchema,
) -> PromptTemplate:
    """
    Develops a parallelizable prompt template for generating a SELECT query with a WHERE clause, given a table schema and property information.
    This prompt focuses on constructing a query that filters data based on properties matching user information, tailored to the specific table schema.
    Parameters:
        table_schema (TableSchema): The schema of the table for query generation.
    Returns:
        PromptTemplate: The constructed prompt template for parallelizable SELECT query generation with a WHERE clause.
    """
    prompt = (
        get_select_query_with_where_text()
        + f"""

    {table_schema}
    
    PROPERTY INFO:

    {{PropertyInfo}}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def build_select_query_with_where_and_top_rows() -> PromptTemplate:
    """
    Constructs a prompt for generating a SELECT query with a WHERE clause using table schema, example rows, and user info.
    This prompt helps create a CQL query that utilizes user-related property values in WHERE clauses for more relevant data retrieval.
    Returns:
        PromptTemplate: The prompt template for SELECT query generation with WHERE clauses.
    """
    prompt = (
        get_select_query_with_where_text()
        + """

    {TableSchema}
    
    {Top3Rows}

    {PropertyInfo}

    RESPONSE:
"""
    )
    return PromptTemplate.from_template(prompt)


def determine_failure_type() -> PromptTemplate:
    """
    Creates a prompt to categorize the type of error encountered in a failed query.
    This prompt helps identify if the failure was due to token count exceeded, syntax error, or other reasons.
    Returns:
        PromptTemplate: The prompt template for error categorization.
    """
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
    """
    Develops a prompt for rewriting a CQL query based on provided error information.
    This prompt assists in correcting and improving a previously failed query by addressing the identified issues.
    Returns:
        PromptTemplate: The prompt template for rewriting a failed query.
    """
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
    """
    Constructs a prompt to determine if any table column name matches user info property names.
    This prompt is used to identify if a table schema is relevant to the user by matching column names with user properties.
    Returns:
        PromptTemplate: The prompt template for matching user properties with table column names.
    """
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


def clean_string_v2(s: str) -> str:
    """
    Cleans a given string by removing backticks, trimming whitespace, and excluding lines containing only 'sql', 'json', or 'cql'.
    This function prepares strings for processing by removing unnecessary elements.
    Parameters:
        s (str): The string to be cleaned.
    Returns:
        str: The cleaned string.
    """
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


### Below prompt templates are useful for
def build_company_description_data() -> PromptTemplate:
    prompt = (
        "You're a helpful assistant and an experienced data expert and business consultant. "
        "From the given company mission statement, I want you to generate a description of what kind of data "
        "they likely will need to have in their business. I want you to generate a hypothetical customer profile "
        "and describe what you would know (or would want to know) about such a customer. "
        "I'm particularly interested in the data that you would store based on your interactions with this customer. "
        "Also, if you can identify some example challenges the customer might experience, please generate a "
        "hypothetical description of an example challenge they might experience. "
        "Return only the requested descriptions without any other narration or explanation."
        ""
        "MISSION STATEMENT:"
        ""
        "{MissionStatement}"
    )
    return PromptTemplate.from_template(prompt)


def build_example_create_table_statements() -> PromptTemplate:
    prompt = (
        "You're a helpful assistant. Don't give me any summary information, explanation, or introduction. "
        "In fact, don't say anything other than what I specify. You're also a great NoSQL database architect. "
        "From the given business description, I want you to generate a list of valid CREATE TABLE statements "
        "for Cassandra to represent customer data. For simplicity, use only TEXT and numeric types, and "
        "don't use UUID or TIMESTAMP. All of the tables should include a user_id column in the primary key. "
        "Return the results as a JSON array of strings."
        ""
        ""
        "BUSINESS DESCRIPTION:"
        ""
        "{BusinessDescription}"
    )

    return PromptTemplate.from_template(prompt)


def build_example_insert_statements() -> PromptTemplate:
    prompt = (
        "You're a helpful assistant. Don't give me any summary information, explanation, or introduction. "
        "In fact, don't say anything other than what I specify. Based on the given business description and "
        "create table statements, generate INSERT statements that generate data for the tables. The data should be "
        "realistic and consistent with the given business description. Return the results as a JSON array of INSERT statements. "
        "Furthermore, ensure that the user_id values match across insert statements so that we can create a complete set of data "
        "for a single user. Create enough rows to paint a complete picture of the given user. Be highly descriptive and verbose "
        "in the data you generate to ensure that the data is as realistic as possible."
        ""
        ""
        "CREATE TABLE STATEMENTS:"
        ""
        "{CreateTableStatements}"
        ""
        ""
        ""
        "BUSINESS DESCRIPTION:"
        ""
        "{BusinessDescription}"
        ""
    )

    return PromptTemplate.from_template(prompt)

def build_questions_from_user_summary() -> PromptTemplate:
    prompt = (
            get_helpful_assistant_prefix()
            + "I will give you a summary of what we know about a customer's interactions with my business. "
              "I want you to formulate a list of {QuestionCount} example questions that the customer might want to ask our public documentation. "
              "Please ensure the questions you generate are relevant to their current question. If no current "
              "question or message is provided by the customer, try your best based on the information I have provided."
              " Return the list of questions as a valid JSON array. "
              ""
              "Here is the summary:"
              ""
              "{CustomerSummary}"
              ""
              ""
              "The customer's current message is:"
              "{UserMessage}"
    )

    return PromptTemplate.from_template(prompt)

def rag_qa_chain() -> PromptTemplate:
    prompt = """
            Answer the customer's question based only on the information provided. If you don't know, then just say "Not answered in content"
            Provide a detailed answer.
            CONTEXT from knowledge base articles:
            {SearchResults}

            QUESTION: {Question}

            YOUR ANSWER:"""
    return PromptTemplate.from_template(prompt)
def build_training_qa_pairs() -> PromptTemplate:
    prompt = """You're a helpful assistant. Do exactly what I say, and don't explain or summarize. 
Generate a JSON array with example question-answer pairs from the following information. Return each pair in the format:
[ {{"question": "Question1?", "answer": "Answer1"}}, {{"question": "Question2?", "answer": "Answer2"}}, . . . ]
Only give one answer for each question. Ensure that each object has exactly one question and one answer.

{Chunk}
"""
    return PromptTemplate.from_template(prompt)