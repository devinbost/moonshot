import asyncio
import logging

from cassandra.cluster import (
    Cluster,
    Session,
)
from cassandra.query import dict_factory
from typing import List, Dict, Tuple, Any, Callable
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
import pandas as pd
from sentence_transformers import SentenceTransformer
import core.ClassInspector
from core.CollectionManager import CollectionManager
from core.ConfigLoader import ConfigLoader
from core.EmbeddingManager import EmbeddingManager
from core.VectorStoreFactory import VectorStoreFactory
from pydantic_models.ComponentData import ComponentData
from langchain.vectorstores import AstraDB

from pydantic_models.TableDescription import TableDescription
from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.TableSchema import TableSchema


class DataAccess:
    # def __init__(self) -> None:
    #     """
    #     Initialize the DataAccess class, setting up various configurations for environment variables,
    #     embeddings, database, and API.
    #     """
    #     self.vector_store = None
    #     self.output_variables = ["new"]
    #     self.data_map = {}
    #     self._initialize_environment_variables()
    #     self._initialize_embeddings()
    #     self._initialize_database_configuration()
    #     self.astrapy_db = AstraPyDB(token=self.token, api_endpoint=self.api_endpoint)
    def __init__(
        self,
        config_loader: ConfigLoader,
        embedding_manager: EmbeddingManager,
        vector_store_factory: VectorStoreFactory,
    ):
        """
        Initialize the DataAccess class.

        This constructor initializes the DataAccess class with necessary configuration,
        embedding management, and vector store factory instances. It sets up the AstraPyDB
        connection using the vector store factory and creates a CollectionManager instance.
        It also initializes output variables and a data map for further operations.

        Parameters:
        - config_loader (ConfigLoader): An instance of ConfigLoader to load necessary configurations.
        - embedding_manager (EmbeddingManager): An instance of EmbeddingManager to manage embeddings used in the class.
        - vector_store_factory (VectorStoreFactory): An instance of VectorStoreFactory to create and manage vector stores.

        The method should be used when an instance of DataAccess is required. It ensures that all
        necessary components for data access are properly initialized and configured.
        """
        self.config_loader = config_loader
        self.embedding_manager = embedding_manager
        self.vector_store_factory = vector_store_factory
        self.astrapy_db = self.vector_store_factory.create_vector_store("AstraPyDB")
        self.collection_manager = CollectionManager(
            self.astrapy_db, self.embedding_manager
        )
        self.output_variables = ["new"]
        self.data_map = {}

    def _initialize_embeddings(self) -> None:
        """
        Initialize the embeddings model and configure the sentence transformer and text splitter for handling text data.
        """
        self.embedding_model = "all-MiniLM-L12-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.embedding_direct = SentenceTransformer(
            "sentence-transformers/" + self.embedding_model
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )

    def getCqlSession(self) -> Session:
        """
        Create and return a Cassandra Query Language (CQL) session with the configured secure bundle and authentication.
        """
        cluster = self.vector_store_factory.create_vector_store("Cluster")
        astra_session = cluster.connect()
        return astra_session

    def setupVectorStoreNew(self, collection: str) -> AstraDB:
        """
        Set up a new vector store in AstraDB with the specified collection name and configured API endpoint and token.
        Parameters:
            collection (str): The name of the collection for the vector store.
        Returns:
            AstraDB: An instance of AstraDB configured with the specified collection.
        """
        return self.vector_store_factory.create_vector_store(
            "AstraDB", collection_name=collection
        )

    def getVectorStore(self, table_name: str) -> Cassandra:
        """
        Retrieve a vector store instance for a specified table name by initializing a new vector store in AstraDB.
        Parameters:
            table_name (str): The name of the table for which the vector store is to be retrieved.
        Returns:
            Cassandra: A Cassandra vector store instance for the specified table name.
        """
        return self.setupVectorStoreNew(table_name)

    def get_output_variable_names(self) -> List[str]:
        """
        Retrieve a list of output variable names that have been added to the DataAccess instance.
        Returns:
            list: A list of output variable names.
        """
        return self.output_variables

    def add_output_variable(self, variable: str) -> None:
        """
        Add a new output variable to the list of output variables in the DataAccess instance.
        Parameters:
            variable (str): The name of the output variable to add.
        """
        self.output_variables.append(variable)

    def add_component(self, component_data: ComponentData) -> None:
        """
        Add a component to the data map with its associated output variable, if present.
        Parameters:
            component_data (ComponentData): The component data to be added.
        """
        self.data_map[component_data.id] = component_data
        if component_data.output_var is not None:
            self.add_output_variable(component_data.output_var)

    def get_data_map(self) -> Dict[str, ComponentData]:
        """
        Retrieve the current data map containing components and their data.
        Returns:
            Dict[str, ComponentData]: A dictionary representing the data map.
        """
        return self.data_map

    def build_graph(
        self, component_dict: Dict[str, ComponentData], graph: Digraph
    ) -> Digraph:
        """
        Build and return a graph visualization from a dictionary of components, adding nodes and edges based on component relationships.
        Parameters:
            component_dict (Dict[str, ComponentData]): A dictionary of component data.
            graph (Digraph): A Graphviz Digraph instance to which the graph will be added.
        Returns:
            Digraph: The updated Graphviz Digraph with the added components and relationships.
        """
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

    def text_after_last_dot(self, input_string: str) -> str:
        """
        Extract and return the substring after the last dot in the input string. Returns an empty string if no dot is found.
        Parameters:
            input_string (str): The input string to process.
        Returns:
            str: The substring after the last dot, or an empty string if no dot is found.
        """
        # Split the string by '.'
        parts = input_string.rsplit(".", 1)

        # Check if there is at least one '.' in the string
        if len(parts) > 1:
            return parts[1]
        else:
            return ""

    def exec_cql_query_simple(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a simple CQL query without specifying a keyspace and return the results as a list of dictionaries.
        Parameters:
            query (str): The CQL query to execute.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the query results.
        """
        print(query)
        session: Session = self.getCqlSession()
        query_stmt = SimpleStatement(query)
        session.row_factory = dict_factory
        rows: List[dict] = session.execute(query_stmt).all()
        print(rows)
        return rows

    async def exec_cql_query_simple_async(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a simple CQL query without specifying a keyspace and return the results as a list of dictionaries.
        Parameters:
            query (str): The CQL query to execute.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the query results.
        """
        print(query)
        session: Session = self.getCqlSession()
        query_stmt = SimpleStatement(query)
        session.row_factory = dict_factory
        response_future = session.execute_async(query_stmt)
        try:
            rows = response_future.result()
            # Await the completion of the query
            return (
                rows.current_rows
            )  # Use the current_rows attribute to get the results
        except Exception as e:
            logging.error(f"Error executing async CQL query: {e}")
            return []

    def get_cql_table_columns(self, table_schema: TableSchema) -> List[ColumnSchema]:
        """
        Retrieve and return a list of column schemas for the specified CQL table schema.
        Parameters:
            table_schema (TableSchema): The schema of the table for which column schemas are to be retrieved.
        Returns:
            List[ColumnSchema]: A list of column schemas for the specified table.
        """
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
            # if item["kind"] == "regular"
        ]
        return table_columns

    async def get_cql_table_columns_async(
        self, table_schema: TableSchema
    ) -> List[ColumnSchema]:
        """
        Retrieve and return a list of column schemas for the specified CQL table schema.
        Parameters:
            table_schema (TableSchema): The schema of the table for which column schemas are to be retrieved.
        Returns:
            List[ColumnSchema]: A list of column schemas for the specified table.
        """
        query = f"SELECT * FROM system_schema.columns WHERE keyspace_name = '{table_schema.keyspace_name}' AND table_name = '{table_schema.table_name}';"
        output = await self.exec_cql_query_simple_async(query)
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
            # if item["kind"] == "regular"
        ]
        return table_columns

    def get_cql_table_indexes(self, table_schema: TableSchema) -> List[str]:
        """
        Retrieve and return a list of index names for the specified CQL table schema.
        Parameters:
            table_schema (TableSchema): The schema of the table for which index names are to be retrieved.
        Returns:
            List[str]: A list of index names for the specified table.
        """
        output = self.exec_cql_query_simple(
            f"SELECT * FROM system_schema.indexes WHERE keyspace_name = '{table_schema.keyspace_name}' AND table_name = '{table_schema.table_name}';"
        )
        names = [idx["index_name"] for idx in output]
        return names

    async def get_cql_table_indexes_async(self, table_schema: TableSchema) -> List[str]:
        """
        Retrieve and return a list of index names for the specified CQL table schema.
        Parameters:
            table_schema (TableSchema): The schema of the table for which index names are to be retrieved.
        Returns:
            List[str]: A list of index names for the specified table.
        """
        output = self.exec_cql_query_simple(
            f"SELECT * FROM system_schema.indexes WHERE keyspace_name = '{table_schema.keyspace_name}' AND table_name = '{table_schema.table_name}';"
        )
        names = [idx["index_name"] for idx in output]
        return names

    def get_table_schemas_in_db_v2(self, empty: str) -> List[TableSchema]:
        """
        Retrieve and return a list of table schemas for all tables in the connected Cassandra database, excluding system tables.
        Parameters:
            empty (str): A string parameter (unused in the current implementation).
        Returns:
            List[TableSchema]: A list of table schemas for non-system tables in the database.
        """
        print(empty)
        session: Session = self.getCqlSession()
        table_entries = session.execute(
            "SELECT keyspace_name, table_name FROM system_schema.tables;"
        )
        table_schemas: List[TableSchema] = []
        for row in table_entries:
            table_name: str = row.table_name
            keyspace_name: str = row.keyspace_name
            if "system" not in keyspace_name:
                table_schema = session.execute(
                    f"SELECT * FROM system_schema.columns WHERE keyspace_name = '{keyspace_name}' AND table_name = '{table_name}'"
                )
                columns = [
                    ColumnSchema(
                        column_name=col.column_name,
                        type=col.type,
                        clustering_order=col.clustering_order,
                        kind=col.kind,
                        position=col.position,
                    )
                    for col in table_schema
                ]
                table_schemas.append(
                    TableSchema(
                        table_name=table_name,
                        keyspace_name=keyspace_name,
                        columns=columns,
                    )
                )
        return table_schemas

    async def get_table_schemas_in_db_v2_async(self, empty: str) -> List[TableSchema]:
        """
        Retrieve and return a list of table schemas for all tables in the connected Cassandra database, excluding system tables.
        Parameters:
            empty (str): A string parameter (unused in the current implementation).
        Returns:
            List[TableSchema]: A list of table schemas for non-system tables in the database.
        """
        print(empty)
        session: Session = self.getCqlSession()
        response_future = session.execute_async(
            "SELECT keyspace_name, table_name FROM system_schema.tables;"
        )
        try:
            table_entries = response_future.result()
            # Await the completion of the query
            table_schemas: List[TableSchema] = []
            for row in table_entries.current_rows:  # Use the current_rows attribute
                table_name: str = row.table_name
                keyspace_name: str = row.keyspace_name
                if "system" not in keyspace_name:
                    response_future_columns = session.execute_async(
                        f"SELECT * FROM system_schema.columns WHERE keyspace_name = '{keyspace_name}' AND table_name = '{table_name}'"
                    )
                    columns_result = (
                        response_future_columns.result()
                    )  # Await the completion of the query for columns
                    columns = [
                        ColumnSchema(
                            column_name=col.column_name,
                            type=col.type,
                            clustering_order=col.clustering_order,
                            kind=col.kind,
                            position=col.position,
                        )
                        for col in columns_result.current_rows  # Use the current_rows attribute
                    ]
                    table_schemas.append(
                        TableSchema(
                            table_name=table_name,
                            keyspace_name=keyspace_name,
                            columns=columns,
                        )
                    )
            return table_schemas
        except Exception as e:
            logging.error(f"Error executing async CQL query: {e}")
            return []

    def get_first_three_rows(
        self, table_schemas: List[TableSchema]
    ) -> List[TableSchema]:
        """
        Retrieve and add the first three rows of data to each table schema in the given list of table schemas.
        Parameters:
            table_schemas (List[TableSchema]): A list of table schemas to which the first three rows of data will be added.
        Returns:
            List[TableSchema]: The updated list of table schemas with the first three rows of data added.
        """
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

    # Use the following method only as a fallback in case the LLM can't get this info for some reason
    def get_table_schemas_that_contain_user_properties(
        self, keyspace: str, column_filters: Dict[str, Tuple[Any, Any]]
    ) -> List[str]:
        """
        Retrieve and return table schemas that contain user-defined properties, filtered by the specified keyspace and column filters.
        Parameters:
            keyspace (str): The keyspace in which to search for tables.
            column_filters (Dict[str, Tuple[Any, Any]]): A dictionary of column filters to apply.
        Returns:
            List[str]: A list of schema descriptions for tables containing the specified user properties.
        """
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

    def generate_python_code(self) -> str:
        """
        Generate and return Python code snippets based on the components and their configurations stored in the data map.
        Returns:
            str: A string of generated Python code snippets.
        """
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

    def summarize_relevant_tables(
        self, tables: List[TableDescription], user_messages: str
    ) -> str:
        """
        Summarize and return contents from relevant tables based on user messages and a list of table descriptions.
        Parameters:
            tables (List[TableDescription]): A list of table descriptions.
            user_messages (str): User messages to be used for summarizing the table contents.
        Returns:
            str: Summaries of contents from relevant tables.
        """
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
        Parameters:
            table_descriptions (List[TableDescription]): A list of TableDescriptions objects.
            relevant_columns (List[str]): A list of relevant column names.
        Returns:
            List[TableDescription]: A filtered list of TableDescriptions objects.
        """
        """
        Filters a list of TableDescriptions objects to include only those where the column_name
        is a member of the relevant_columns list.

        :param table_descriptions: List of TableDescriptions objects.
        :param relevant_columns: List of relevant column names.
        :return: Filtered list of TableDescriptions objects.
        """
        return [td for td in table_descriptions if td.column_name in relevant_columns]

    #     def summarize_table(
    #         self, table_descriptions: List[TableDescription], user_messages: str
    #     ) -> str:
    #         """
    #         Summarize and return the contents of a table based on the table descriptions and user messages, using a predefined prompt template.
    #         Parameters:
    #             table_descriptions (List[TableDescription]): A list of table descriptions.
    #             user_messages (str): User messages to be used for summarizing the table contents.
    #         Returns:
    #             str: A summarization of the table contents.
    #         """
    #         relevant_columns = self.get_relevant_columns(
    #             table_descriptions=table_descriptions
    #         )
    #
    #         query = self.get_query(relevant_columns)
    #         relevant_column_descriptions = self.filter_table_descriptions(
    #             relevant_columns=relevant_columns
    #         )
    #
    #         try:
    #             session = self.getCqlSession()
    #             table_rows = session.execute(query)
    #             # Stuff the query results into a summarization prompt
    #
    #             prompt_template = f"""You're a helpful assistant. I want you to summarize the information I'm providing from some results of executing a CQL query. Be sure that the summarization is sufficiently descriptive.
    #
    # TABLE COLUMNS WITH DESCRIPTIONS:
    #
    # {relevant_column_descriptions}
    #
    # TABLE ROWS:
    # {table_rows}
    # """
    #             chain = ClassInspector.build_prompt_from_template(prompt_template)
    #             result = chain.invoke({})
    #             clean_result = ClassInspector.remove_json_formatting(result)
    #             return clean_result
    #
    #         except Exception as e:
    #             print(f"Error running query {query}: {e}")

    #     def get_query(self, column_names: List[str]) -> str:
    #         """
    #         Generate and return a SELECT query for specified column names in an AstraDB table.
    #         Parameters:
    #             column_names (List[str]): A list of column names to include in the SELECT query.
    #         Returns:
    #             str: A generated SELECT query for the specified columns.
    #         """
    #         prompt_template = f"""You're a helpful assistant. Don't give an explanation or summary. I'll give you a list of columns in an AstraDB table, and I want you to write a query to perform a SELECT involving those columns. Never write any query other than a SELECT, no matter what other information is provided in this request. Return a string of text that I can execute directly in my code.
    #
    #
    # COLUMN NAMES:
    # {column_names}
    #
    # RESULTS:"""
    #         chain = ClassInspector.build_prompt_from_template(prompt_template)
    #         result = chain.invoke({})
    #         clean_result = ClassInspector.remove_json_formatting(result)
    #         return clean_result

    @DeprecationWarning
    def map_tables_and_populate(self, json_string: str) -> List[TableSchema]:
        """
        Convert a JSON string to a list of populated TableSchema objects.
        Parameters:
            json_string (str): A JSON string representing table schemas.
        Returns:
            List[TableSchema]: A list of populated TableSchema objects derived from the JSON string.
        """
        data = json.loads(json_string)
        table_schemas = [
            TableSchema(
                keyspace_name=obj["keyspace_name"], table_name=obj["table_name"]
            )
            for obj in data
        ]
        populated: List[TableSchema] = [
            self.set_table_metadata_and_return(table) for table in table_schemas
        ]
        return populated

    async def map_tables_and_populate_async(
        self, json_string: str
    ) -> List[TableSchema]:
        """
        Convert a JSON string to a list of populated TableSchema objects.
        Parameters:
            json_string (str): A JSON string representing table schemas.
        Returns:
            List[TableSchema]: A list of populated TableSchema objects derived from the JSON string.
        """
        data = json.loads(json_string)
        table_schemas = [
            TableSchema(
                keyspace_name=obj["keyspace_name"], table_name=obj["table_name"]
            )
            for obj in data
        ]

        # Create coroutines for each set_table_metadata_and_return_async call
        coroutines = [
            self.set_table_metadata_and_return_async(table) for table in table_schemas
        ]

        # Await all coroutines to complete and return their results
        populated = await asyncio.gather(*coroutines)
        return populated

    #     def get_relevant_columns(
    #         self, table_descriptions: List[TableDescription]
    #     ) -> List[str]:
    #         """
    #         Determine and return a list of relevant column names for a chatbot, based on the provided table descriptions.
    #         Parameters:
    #             table_descriptions (List[TableDescription]): A list of table descriptions.
    #         Returns:
    #             List[str]: A list of column names deemed relevant for a chatbot.
    #         """
    #         prompt_template = f"""You're a helpful assistant. Don't give an explanation or summary. I'll give you the name of a table, along with its columns and their descriptions, and I want you to return a JSON list of the columns that might be helpful for a chatbot. Return only the JSON list that I can execute directly in my code. The JSON list should only contain column_name.
    #
    #
    # TABLE WITH COLUMN DESCRIPTIONS:
    # {table_descriptions}
    #
    # RESULTS:"""
    #         chain = ClassInspector.build_prompt_from_template(prompt_template)
    #         result = chain.invoke({})
    #         clean_result = ClassInspector.remove_json_formatting(result)
    #         columns_as_json = json.loads(clean_result)
    #         column_name_list = [item["column_name"] for item in columns_as_json]
    #         return column_name_list

    def set_table_metadata_and_return(self, table_schema: TableSchema) -> TableSchema:
        """
        Set metadata for the given table schema and return the updated schema.
        Parameters:
            table_schema (TableSchema): The table schema for which metadata is to be set.
        Returns:
            TableSchema: The updated table schema with set metadata.
        """
        indexes = self.get_cql_table_indexes(table_schema)
        columns = self.get_cql_table_columns(table_schema)
        table_schema.indexes = indexes
        table_schema.columns = columns
        return table_schema

    async def set_table_metadata_and_return_async(
        self, table_schema: TableSchema
    ) -> TableSchema:
        """
        Set metadata for the given table schema and return the updated schema.
        Parameters:
            table_schema (TableSchema): The table schema for which metadata is to be set.
        Returns:
            TableSchema: The updated table schema with set metadata.
        """
        indexes = await self.get_cql_table_indexes_async(table_schema)
        columns = await self.get_cql_table_columns_async(table_schema)
        table_schema.indexes = indexes
        table_schema.columns = columns
        return table_schema

    @DeprecationWarning
    def get_path_segment_keywords(self) -> Dict[str, List[str]]:
        """
        Retrieve and return a dictionary of distinct path segment keywords from the default Cassandra keyspace.
        Returns:
            Dict[str, List[str]]: A dictionary where each key is a metadata path segment and the value is a list of distinct keywords.
        """
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
        # distinct_seg1 = df["seg1"].unique().tolist()
        filtered_df = df[
            ~df["seg2"].str.contains("knowledge-base", na=False)
            & ~df["seg2"].str.contains("legal", na=False)
        ]
        distinct_seg2 = [x for x in filtered_df["seg2"].unique() if x is not None]
        distinct_seg3 = [x for x in df["seg3"].unique() if x is not None]
        distinct_seg4 = [x for x in df["seg4"].unique() if x is not None]
        # distinct_seg5 = df["seg5"].unique().tolist()
        # distinct_seg6 = df["seg6"].unique().tolist()
        distinct_values_dict = {
            "metadata.path_segment_2": distinct_seg2,
            "metadata.path_segment_3": distinct_seg3,
            "metadata.path_segment_4": distinct_seg4,
            # "metadata.path_segment_5": distinct_seg5,
            # "metadata.path_segment_6": distinct_seg6,
        }
        return distinct_values_dict

    async def get_path_segment_keywords_async(self) -> Dict[str, List[str]]:
        """
        Retrieve and return a dictionary of distinct path segment keywords from the default Cassandra keyspace.
        Returns:
            Dict[str, List[str]]: A dictionary where each key is a metadata path segment and the value is a list of distinct keywords.
        """
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
        response_future = session.execute_async(query)
        try:
            results = response_future.result()
            # Await the completion of the query
            # Convert results to a DataFrame
            df = pd.DataFrame(results.current_rows)  # Use the current_rows attribute
            # distinct_seg1 = df["seg1"].unique().tolist()
            filtered_df = df[
                ~df["seg2"].str.contains("knowledge-base", na=False)
                & ~df["seg2"].str.contains("legal", na=False)
            ]
            distinct_seg2 = [x for x in filtered_df["seg2"].unique() if x is not None]
            distinct_seg3 = [x for x in df["seg3"].unique() if x is not None]
            distinct_seg4 = [x for x in df["seg4"].unique() if x is not None]
            # distinct_seg5 = df["seg5"].unique().tolist()
            # distinct_seg6 = df["seg6"].unique().tolist()
            distinct_values_dict = {
                "metadata.path_segment_2": distinct_seg2,
                "metadata.path_segment_3": distinct_seg3,
                "metadata.path_segment_4": distinct_seg4,
                # "metadata.path_segment_5": distinct_seg5,
                # "metadata.path_segment_6": distinct_seg6,
            }
            return distinct_values_dict
        except Exception as e:
            logging.error(f"Error executing async CQL query: {e}")
            return {}
