from cassandra.cluster import (
    Cluster,
    Session,
    ResultSet,
)
from cassandra.query import dict_factory
from typing import List, Dict, Tuple, Any
from cassandra.auth import PlainTextAuthProvider
import hashlib
import os
import json
from astrapy.db import AstraDB as AstraPyDB, AstraDBCollection
from cassandra.query import SimpleStatement
from graphviz import Digraph
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Cassandra
import wget
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
import ClassInspector
from Config import config
from pydantic_models.ComponentData import ComponentData
from langchain.vectorstores import AstraDB

from pydantic_models.TableDescription import TableDescription
from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.TableKey import TableKey
from pydantic_models.TableSchema import TableSchema


class DataAccess:
    def __init__(self):
        self.vector_store = None
        self.output_variables = ["new"]
        self.data_map = {}
        self._initialize_environment_variables()
        self._initialize_embeddings()
        self._initialize_database_configuration()
        self._initialize_api_configuration()

    def _initialize_environment_variables(self):
        self.keyspace = os.getenv("KEYSPACE_NAME", "keyspace")
        self.table_name = os.getenv("TABLE_NAME", "table")
        self.database_name = os.getenv("DATABASE_NAME", "database")

    def _initialize_embeddings(self):
        self.embedding_model = "all-MiniLM-L12-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.embedding_direct = SentenceTransformer(
            "sentence-transformers/" + self.embedding_model
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=1500,
            length_function=len,
            is_separator_regex=False,
        )

    def _initialize_database_configuration(self):
        self.secure_bundle_path = config.get_secure_bundle_full_path()
        self.vector_store = self._setupVectorStore()

    def _initialize_api_configuration(self):
        with open(config.openai_json) as f:
            secrets = json.load(f)
        self.token = secrets[
            "token"
        ]  # In production, it's better to only load the secrets as needed (and have them always encrypted except when needed), but this is okay for now.
        self.api_endpoint = secrets["endpoint"]
        self.astrapy_db = AstraPyDB(token=self.token, api_endpoint=self.api_endpoint)

    def getCqlSession(self) -> Session:
        cluster = Cluster(
            cloud={
                "secure_connect_bundle": self.secure_bundle_path,
            },
            auth_provider=PlainTextAuthProvider(
                "token",
                os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"),
            ),
        )

        astra_session = cluster.connect()
        return astra_session

    def setupVectorStoreNew(self, collection: str) -> AstraDB:
        cassandraVectorStore = AstraDB(
            embedding=self.embeddings,
            collection_name=collection,  # Replace with your collection name
            token=os.getenv(
                "ASTRA_DB_TOKEN_BASED_PASSWORD"
            ),  # Replace with your AstraDB token
            api_endpoint=self.api_endpoint,  # Replace with your AstraDB API endpoint
        )
        return cassandraVectorStore

    def _setup_vector_store(self, table_name: str) -> Cassandra:
        return Cassandra(
            embedding=self.embeddings,
            session=self.getCqlSession(),
            keyspace=self.keyspace,
            table_name=table_name,
        )

    def _setupVectorStore(self) -> Cassandra:
        return Cassandra(
            embedding=self.embeddings,
            session=self.getCqlSession(),
            keyspace=self.keyspace,
            table_name=self.table_name,
        )

    def getVectorStore(self, table_name: str) -> Cassandra:
        return self.setupVectorStoreNew(table_name)

    def loadWikipediaData(self):
        url = "https://raw.githubusercontent.com/GeorgeCrossIV/Langchain-Retrieval-Augmentation-with-CASSIO/main/20220301.simple.csv"
        sample_data = config.data_path + "/20220301.simple.csv"

        if not os.path.isfile(sample_data):
            wget.download(url)
        data = pd.read_csv(sample_data)
        data = data.head(10)
        data = data.rename(columns={"text ": "text"})
        for index, row in data.iterrows():
            self._parseWikipediaRow(row)

    def _parseWikipediaRow(self, row):
        metadata = {"url": row["url"], "title": row["title"]}
        page_content = row["text"]

        wikiDocument = Document(page_content=page_content, metadata=metadata)
        wikiDocs = self.splitter.transform_documents([wikiDocument])
        self.vector_store.add_documents(wikiDocs)

    def get_output_variable_names(self):
        return self.output_variables

    def add_output_variable(self, variable):
        self.output_variables.append(variable)

    def add_component(self, component_data: ComponentData):
        self.data_map[component_data.id] = component_data
        if component_data.output_var is not None:
            self.add_output_variable(component_data.output_var)

    def get_data_map(self):
        return self.data_map

    def build_graph(self, component_dict: dict[str, ComponentData], graph: Digraph):
        # Style for nodes
        graph.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="lightgrey",
            fontname="Helvetica",
        )

        # Add nodes with additional attributes
        for left_id, left_data in component_dict.items():
            trimmed_class_name = self.text_after_last_dot(left_data.class_name)
            label = f"{trimmed_class_name}"
            # label = f"{left_data.component_name} | Class: {left_data.class_name} | Library: {left_data.library} | Access: {left_data.access_type} | Params: {left_data.params} | Output: {left_data.output_var}"
            graph.node(left_id, label=label)
            print(f"Adding graph.node({left_id}, label={label})")

        # Add edges
        for left_id, left_data in component_dict.items():
            for right_id, right_data in component_dict.items():
                if left_id == right_id:
                    continue
                if left_data.output_var in right_data.params.values():
                    trimmed_left_class_name = self.text_after_last_dot(
                        left_data.class_name
                    )
                    trimmed_right_class_name = self.text_after_last_dot(
                        right_data.class_name
                    )
                    left_label = f"{trimmed_left_class_name}"
                    right_label = f"{trimmed_right_class_name}"
                    # left_label = f"{left_data.component_name} | Class: {left_data.class_name} | Library: {left_data.library} | Access: {left_data.access_type} | Params: {left_data.params}"
                    # right_label = f"{right_data.component_name} | Class: {right_data.class_name} | Library: {right_data.library} | Access: {right_data.access_type} | Params: {right_data.params}"
                    print(f"left_label is {left_label})")
                    print(f"right_label is {right_label})")
                    graph.edge(left_id, right_id, label=left_data.output_var)
                    print(
                        f"Adding graph.edge({left_id}, {right_id}, label={left_data.output_var})"
                    )
        return graph

    def text_after_last_dot(self, input_string: str):
        """
        Extracts the text to the right of the last '.' character in a string.
        If there is no '.' in the string, it returns an empty string.
        """
        # Split the string by '.'
        parts = input_string.rsplit(".", 1)

        # Check if there is at least one '.' in the string
        if len(parts) > 1:
            return parts[1]
        else:
            return ""

    def exec_cql_query(self, keyspace: str, query: str) -> List[dict]:
        """Remember, we need to sanitize this!"""
        session: Session = self.getCqlSession()
        session.set_keyspace(keyspace)
        query_stmt = SimpleStatement(query)
        session.row_factory = dict_factory
        rows: List[dict] = session.execute(query_stmt).all()
        return rows

    def exec_cql_query_simple(self, query: str) -> List[dict]:
        """Remember, we need to sanitize this!"""
        session: Session = self.getCqlSession()
        query_stmt = SimpleStatement(query)
        session.row_factory = dict_factory
        rows: List[dict] = session.execute(query_stmt).all()
        return rows

    def get_cql_table_description(self, table_schema: TableSchema) -> str:
        output = self.exec_cql_query_simple(
            f"describe {table_schema.keyspace_name}.{table_schema.table_name};"
        )
        return output

    def get_cql_table_columns(self, table_schema: TableSchema) -> List[ColumnSchema]:
        query = f"SELECT * FROM system_schema.columns WHERE keyspace_name = '{table_schema.keyspace_name}' AND table_name = '{table_schema.table_name}';"
        output = self.exec_cql_query_simple(query)
        table_columns = [
            ColumnSchema(
                **{
                    key: item[key]
                    for key in [
                        "column_name",
                        "clustering_order",
                        "kind",
                        "position",
                        "type",
                    ]
                }
            )
            for item in output
            if item["kind"] != "regular"
        ]
        return table_columns

    def get_cql_table_indexes(self, table_schema: TableSchema) -> list[str]:
        output = self.exec_cql_query_simple(
            f"SELECT * FROM system_schema.indexes WHERE keyspace_name = '{table_schema.keyspace_name}' AND table_name = '{table_schema.table_name}';"
        )
        names = [idx["index_name"] for idx in output]
        return names

    def get_table_schemas_in_db(self) -> List[TableSchema]:
        session: Session = self.getCqlSession()
        rows = session.execute("SELECT table_name FROM system_schema.tables;")
        table_schemas: List[TableSchema] = []
        for row in rows:
            table_name: str = row.table_name
            table_schema = session.execute(
                f"SELECT * FROM system_schema.columns WHERE keyspace_name = '{keyspace}' AND table_name = '{table_name}'"
            )
            columns = [
                ColumnSchema(column_name=col.column_name, column_type=col.type)
                for col in table_schema
            ]
            table_schemas.append(
                TableSchema(
                    table_name=table_name, keyspace_name=keyspace, columns=columns
                )
            )
        return table_schemas

    def get_table_schemas(self, keyspace: str) -> List[TableSchema]:
        session: Session = self.getCqlSession()
        session.set_keyspace(keyspace)
        rows = session.execute(
            "SELECT table_name FROM system_schema.tables WHERE keyspace_name = %s",
            [keyspace],
        )
        table_schemas: List[TableSchema] = []
        for row in rows:
            table_name: str = row.table_name
            table_schema = session.execute(
                f"SELECT * FROM system_schema.columns WHERE keyspace_name = '{keyspace}' AND table_name = '{table_name}'"
            )
            columns = [
                ColumnSchema(column_name=col.column_name, column_type=col.type)
                for col in table_schema
            ]
            table_schemas.append(
                TableSchema(
                    table_name=table_name, keyspace_name=keyspace, columns=columns
                )
            )
        return table_schemas

    def get_first_three_rows(
        self, table_schemas: List[TableSchema]
    ) -> List[TableSchema]:
        session: Session = self.getCqlSession()
        for table_schema in table_schemas:
            query = f"SELECT * FROM {table_schema.keyspace_name}.{table_schema.table_name} LIMIT 3"
            rows = session.execute(query)
            table_schema.rows = [
                {
                    col.column_name: getattr(row, col.column_name)
                    for col in table_schema.columns
                }
                for row in rows
            ]
        return table_schemas

    def get_table_schemas_in_db(self) -> List[TableSchema]:
        output = self.exec_cql_query_simple(
            "SELECT keyspace_name, table_name FROM system_schema.tables"
        )
        tables = [
            TableSchema(**{key: item[key] for key in ["keyspace_name", "table_name"]})
            for item in output
            if "system" not in item["keyspace_name"]
        ]
        for table in tables:
            self.set_table_metadata(table)
        return tables

    # Use the following method only as a fallback in case the LLM can't get this info for some reason
    def get_table_schemas_that_contain_user_properties(
        self, keyspace: str, column_filters: Dict[str, Tuple[Any, Any]]
    ) -> List[str]:
        session: Session = self.getCqlSession()
        session.set_keyspace(keyspace)
        rows = session.execute(
            "SELECT table_name FROM system_schema.tables WHERE keyspace_name = %s",
            [keyspace],
        )
        table_schemas: List[str] = []

        for row in rows:
            table_name: str = row.table_name
            table_schema = session.execute(
                f"SELECT * FROM system_schema.columns WHERE keyspace_name = '{keyspace}' AND table_name = '{table_name}'"
            )

            include_table = False
            schema_description: str = f"Table: {table_name}\n"

            for col in table_schema:
                if col.column_name in column_filters:
                    include_table = True
                    schema_description += f"{col.column_name} {col.type}\n"

            if include_table:
                table_schemas.append(schema_description.strip())

        return table_schemas

    def generate_python_code(self):
        code_snippets = []

        for component_id, component_data in self.data_map.items():
            code_line = ""

            # Handling class construction
            if component_data.access_type == "constructor":
                params_str = ", ".join(
                    [f"{k}={v}" for k, v in component_data.params.items()]
                )
                code_line = f"{component_data.output_var} = {component_data.library}.{component_data.class_name}({params_str})"

            # Handling method calls
            elif component_data.access_type == "method":
                params_str = ", ".join(
                    [f"{k}={v}" for k, v in component_data.params.items()]
                )
                if component_data.output_var:
                    code_line = f"{component_data.output_var} = {component_data.class_name}.{component_data.component_name}({params_str})"
                else:
                    code_line = f"{component_data.class_name}.{component_data.component_name}({params_str})"

            # Handling property access
            elif component_data.access_type == "property":
                if component_data.params:
                    # Assuming property setting
                    for prop, value in component_data.params.items():
                        code_line = f"{component_data.class_name}.{prop} = {value}"
                else:
                    # Assuming property getting
                    code_line = f"{component_data.output_var} = {component_data.class_name}.{component_data.component_name}"

            # Add the generated line to the snippets list
            if code_line:
                code_snippets.append(code_line)

        return "\n".join(code_snippets)

    def save_prompt(self, prompt: str):
        mycollections = self.astrapy_db.get_collections()["status"]["collections"]
        if "prompts" not in mycollections:
            collection = self.astrapy_db.create_collection(
                collection_name="prompts", dimension=384
            )
        else:
            collection = AstraDBCollection(
                collection_name="prompts", astra_db=self.astrapy_db
            )
        # Workaround due to strange uuid bug:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        vector = self.embedding_direct.encode(prompt).tolist()
        collection.insert_one({"_id": prompt_hash, "prompt": prompt, "$vector": vector})

    def get_matching_prompts(self, match: str):
        mycollections = self.astrapy_db.get_collections()["status"]["collections"]
        if "prompts" not in mycollections:
            collection = self.astrapy_db.create_collection(
                collection_name="prompts", dimension=384
            )
        else:
            collection = AstraDBCollection(
                collection_name="prompts", astra_db=self.astrapy_db
            )
        vector = self.embedding_direct.encode(match).tolist()
        results = collection.vector_find(vector, limit=10)
        return results
        # Query DB for prompts

    def get_user_profile(
        self, table_names: List[str], phone_number: str
    ) -> List[TableDescription]:
        combined_results = []
        session = self.getCqlSession()
        for table_name in table_names:
            query = f"SELECT * FROM {table_name} WHERE phone_number = %s"
            try:
                rows = session.execute(query, [phone_number])
                for row in rows:
                    combined_results.append(
                        TableDescription(table_name=table_name, **row)
                    )
            except Exception as e:
                print(f"Error querying table {table_name}: {e}")
        return combined_results

    def get_relevant_tables(self, tables: List[TableDescription], user_messages: str):
        prompt_template = f"""You're a helpful assistant. Don't give an explanation or summary. I'll give you a list of tables, columns and descriptions, and I want you to determine which tables are most likely to be relevant to the user's request. Then, return them as a JSON list. 

TABLES:
{tables}


USER INFORMATION:

{user_messages}
RESULTS:
"""
        chain = ClassInspector.build_prompt_from_template(prompt_template)
        result = chain.invoke({})
        clean_result = ClassInspector.remove_json_formatting(result)
        result_as_json = json.loads(clean_result)
        return clean_result

    def summarize_relevant_tables(
        self, tables: List[TableDescription], user_messages: str
    ):
        formatted_summaries = []

        # Need to parallelize the following method:
        def process_table(table: str):
            summarization = self.summarize_table(
                table_descriptions=tables, user_messages=user_messages
            )
            summarization_template = (
                f"\n\n\nSUMMARY OF CONTENTS FROM TABLE {table}: \n\n{summarization}"
            )
            return summarization_template

        for t in tables:
            process_table(t.table_name)
        # with ThreadPoolExecutor() as executor:
        #     formatted_summaries = list(executor.map(process_table, tables))

        combined_summaries = "\n".join(formatted_summaries)
        return combined_summaries

    def filter_table_descriptions(
        self, table_descriptions: List[TableDescription], relevant_columns: List[str]
    ) -> List[TableDescription]:
        """
        Filters a list of TableDescriptions objects to include only those where the column_name
        is a member of the relevant_columns list.

        :param table_descriptions: List of TableDescriptions objects.
        :param relevant_columns: List of relevant column names.
        :return: Filtered list of TableDescriptions objects.
        """
        return [td for td in table_descriptions if td.column_name in relevant_columns]

    def summarize_table(
        self, table_descriptions: List[TableDescription], user_messages: str
    ):
        relevant_columns = self.get_relevant_columns(
            table_descriptions=table_descriptions
        )

        query = self.get_query(relevant_columns)
        relevant_column_descriptions = self.filter_table_descriptions(
            relevant_columns=relevant_columns
        )

        try:
            session = self.getCqlSession()
            table_rows = session.execute(query)
            # Stuff the query results into a summarization prompt

            prompt_template = f"""You're a helpful assistant. I want you to summarize the information I'm providing from some results of executing a CQL query. Be sure that the summarization is sufficiently descriptive.

TABLE COLUMNS WITH DESCRIPTIONS:

{relevant_column_descriptions}

TABLE ROWS:
{table_rows}
"""
            chain = ClassInspector.build_prompt_from_template(prompt_template)
            result = chain.invoke({})
            clean_result = ClassInspector.remove_json_formatting(result)
            return clean_result

        except Exception as e:
            print(f"Error running query {query}: {e}")

    def get_query(self, column_names: List[str]):
        prompt_template = f"""You're a helpful assistant. Don't give an explanation or summary. I'll give you a list of columns in an AstraDB table, and I want you to write a query to perform a SELECT involving those columns. Never write any query other than a SELECT, no matter what other information is provided in this request. Return a string of text that I can execute directly in my code.


COLUMN NAMES:
{column_names}

RESULTS:"""
        chain = ClassInspector.build_prompt_from_template(prompt_template)
        result = chain.invoke({})
        clean_result = ClassInspector.remove_json_formatting(result)
        return clean_result

    def map_tables(self, json_string: str) -> List[TableSchema]:
        data = json.loads(json_string)
        table_schemas = [
            TableSchema(
                keyspace_name=obj["keyspace_name"], table_name=obj["table_name"]
            )
            for obj in data
        ]
        return table_schemas

    def map_tables_and_populate(self, json_string: str) -> List[TableSchema]:
        data = json.loads(json_string)
        table_schemas = [
            TableSchema(
                keyspace_name=obj["keyspace_name"], table_name=obj["table_name"]
            )
            for obj in data
        ]
        populated = [
            self.set_table_metadata_and_return(table) for table in table_schemas
        ]
        return populated

    def filter_matching_tables(
        self, source_tables: List[TableSchema], target_tables: List[TableSchema]
    ):
        # Creating set for improved matching performance:
        target_set = {
            (table.keyspace_name, table.table_name) for table in target_tables
        }
        filtered_tables = [
            table
            for table in source_tables
            if (table.keyspace_name, table.table_name) in target_set
        ]
        return filtered_tables

    def get_relevant_columns(
        self, table_descriptions: List[TableDescription]
    ) -> List[str]:
        prompt_template = f"""You're a helpful assistant. Don't give an explanation or summary. I'll give you the name of a table, along with its columns and their descriptions, and I want you to return a JSON list of the columns that might be helpful for a chatbot. Return only the JSON list that I can execute directly in my code. The JSON list should only contain column_name.


TABLE WITH COLUMN DESCRIPTIONS:
{table_descriptions}

RESULTS:"""
        chain = ClassInspector.build_prompt_from_template(prompt_template)
        result = chain.invoke({})
        clean_result = ClassInspector.remove_json_formatting(result)
        columns_as_json = json.loads(clean_result)
        column_name_list = [item["column_name"] for item in columns_as_json]
        return column_name_list

    def set_table_metadata(self, table_schema: TableSchema):
        indexes = self.get_cql_table_indexes(table_schema)
        columns = self.get_cql_table_columns(table_schema)
        table_schema.indexes = indexes
        table_schema.columns = columns

    def set_table_metadata_and_return(self, table_schema: TableSchema):
        indexes = self.get_cql_table_indexes(table_schema)
        columns = self.get_cql_table_columns(table_schema)
        table_schema.indexes = indexes
        table_schema.columns = columns
        return table_schema

    def get_path_segment_keywords(self):
        query = SimpleStatement(
            f"""SELECT query_text_values['metadata.subdomain'] as subdomain,
         query_text_values['metadata.path_segment_1'] as seg1,
         query_text_values['metadata.path_segment_2'] as seg2,
         query_text_values['metadata.path_segment_3'] as seg3,
         query_text_values['metadata.path_segment_4'] as seg4,
         query_text_values['metadata.path_segment_5'] as seg5,
         query_text_values['metadata.path_segment_6'] as seg6,
         query_text_values['metadata.title'] as title,
         query_text_values['metadata.nlp_keywords'] as keywords
         FROM default_keyspace.sitemapls;"""
        )
        session = self.getCqlSession()
        # execute the query
        session.default_timeout = 120
        results = session.execute(query)

        # Convert results to a DataFrame
        df = pd.DataFrame(results)
        distinct_seg1 = df["seg1"].unique()
        filtered_df = df[~df["seg2"].str.contains("knowledge-base", na=False)]
        distinct_seg2 = filtered_df["seg2"].unique()
        distinct_seg3 = df["seg3"].unique()
        distinct_seg4 = df["seg4"].unique()
        distinct_seg5 = df["seg5"].unique()
        distinct_seg6 = df["seg6"].unique()
        distinct_values_dict = {
            "seg1": distinct_seg1,
            "seg2": distinct_seg2,
            "seg3": distinct_seg3,
            "seg4": distinct_seg4,
            "seg5": distinct_seg5,
            "seg6": distinct_seg6,
        }
        return distinct_values_dict

    def filtered_ANN_search(
        self, collection_filter: dict[str, str], user_summary: dict[str, str]
    ):
        user_summary_string = json.dumps(user_summary)
        input_vector = self.embedding_direct.encode(user_summary_string).tolist()
        collection = AstraDBCollection(
            collection_name="sitemapls", astra_db=self.astrapy_db
        )
        results = collection.vector_find(
            vector=input_vector,
            filter=collection_filter,
            limit=100,
        )
        return results


def get_distinct_path_segments(session, segment_key):
    # Adjust the query to use the specified segment key
    query = SimpleStatement(
        f"""SELECT query_text_values['metadata.{segment_key}'] FROM default_keyspace.sitemapls;"""
    )
    new_results = session.execute(query)
    rows = new_results.all()

    # Extract the distinct values using a set comprehension
    distinct_values = {
        getattr(row, f"query_text_values__metadata_{segment_key}") for row in rows
    }
    return list(distinct_values)
