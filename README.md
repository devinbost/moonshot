# General Info
The purpose of this repo is to use AstraDB with a web crawler to extract code and execute with LangChain and an LLM

Start from UserInterface.py

Most easily navigated in PyCharm

Reach out to Devin for more info if interested.

## Setup
Setup your PyCharm to run Streamlit like this:
![img.png](img%2Fimg.png)

The following parameters are currently configurable as environment variables:
```commandline
PYTHONUNBUFFERED=1;
ASTRA_DB_TOKEN_BASED_PASSWORD=AstraCS:...;
IBM_API_SECRET=...;
IBM_PROJECT_ID=...;
KEYSPACE=openai;
OPENAI_API_KEY=...;
PROVIDER=OPENAI;
TABLE_NAME=mytable;
```