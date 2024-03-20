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
import ClassInspector
from core.CollectionManager import CollectionManager
from core.ConfigLoader import ConfigLoader
from core.VectorStoreFactory import VectorStoreFactory
from pydantic_models.ComponentData import ComponentData
from langchain.vectorstores import AstraDB

from pydantic_models.TableDescription import TableDescription
from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.TableSchema import TableSchema


class DataAccess:
    def __init__(
        self,
        config_loader: ConfigLoader,
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
        self.vector_store_factory = vector_store_factory
        self.output_variables = ["new"]
        self.data_map = {}

    def getCqlSession(self) -> Session:
        """
        Create and return a Cassandra Query Language (CQL) session with the configured secure bundle and authentication.
        """
        cluster = self.vector_store_factory.create_vector_store("Cluster")
        astra_session = cluster.connect()
        return astra_session

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
