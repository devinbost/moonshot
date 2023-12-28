from langchain.prompts import PromptTemplate

from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo


def get_python_code_gen_prefix():
    return f"""You're a helpful assistant. I will give you a scenario, and I want you to generate Python code that performs the task. Don't give me any summary information, explanation, or introduction. In fact, don't say anything other than the code. \n"""


def get_relevant_user_tables(tables: list[TableSchema], user_info: UserInfo):
    prefix = get_python_code_gen_prefix()
    prompt = f"""{prefix} \n I will give you a list of table schemas and some user info, and I want you to return the names of tables that have any column name that matches any of the user info property names."""
    for table in tables:
        prompt += table.to_lcel_json()


def create_synopsis_prompt(title):
    prefix = get_python_code_gen_prefix()
    prompt = PromptTemplate.from_template(
        f"""{prefix}
        I will give you some examples from docs: 

EXAMPLES:

{title}
Playwright: This is a synopsis for the above play:"""
    )
