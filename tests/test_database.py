import os
import unittest
from unittest.mock import MagicMock, patch

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, Session
from cassandra.query import SimpleStatement

from DataAccess import DataAccess
import uuid

from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.TableKey import TableKey
from pydantic_models.TableSchema import TableSchema
from Config import config


class TestGetTableSchemas(unittest.TestCase):
    def test_populate_indexes(self):
        fake_data_access = DataAccess()
        table_schema = TableSchema(
            keyspace_name="telecom",
            table_name="customer_support_transcripts",
            columns=[
                ColumnSchema(column_name="id", column_type="int"),
                ColumnSchema(column_name="name", column_type="string"),
            ],
        )
        indexes = fake_data_access.get_cql_table_indexes(table_schema)

        self.assertEqual(
            [
                "customer_support_transcripts_issue_type_idx",
                "customer_support_transcripts_resolution_status_idx",
            ],
            indexes,
        )

    def test_populate_keys(self):
        fake_data_access = DataAccess()
        table_schema = TableSchema(
            keyspace_name="telecom",
            table_name="customer_support_transcripts",
            columns=[
                ColumnSchema(column_name="id", column_type="int"),
                ColumnSchema(column_name="name", column_type="string"),
            ],
        )
        indexes = fake_data_access.get_cql_table_columns(table_schema)
        expected_keys = [
            TableKey(
                column_name="phone_number",
                clustering_order="none",
                kind="partition_key",
                position=0,
            ),
            TableKey(
                column_name="transcript_id",
                clustering_order="asc",
                kind="clustering",
                position=0,
            ),
        ]

        self.assertEqual(expected_keys, indexes)

    @patch("DataAccess.DataAccess.getCqlSession")
    def test_get_table_schemas(self, mock_get_session):
        # Mock data
        keyspace = "telecom"
        mock_tables = [
            {"table_name": "customer_support_transcripts"},
            {"table_name": "family_plan_info"},
        ]

        mock_columns_customer_support_transcripts = [
            {"column_name": "phone_number", "type": "text"},
            {"column_name": "transcript_id", "type": "uuid"},
            {"column_name": "customer_name", "type": "text"},
            {"column_name": "interaction_date", "type": "timestamp"},
            {"column_name": "issue_type", "type": "text"},
            {"column_name": "resolution_status", "type": "text"},
            {"column_name": "transcript", "type": "text"},
        ]

        mock_columns_family_plan_info = [
            {"column_name": "phone_number", "type": "text"},
            {"column_name": "family_member_phone_number", "type": "text"},
            {"column_name": "age", "type": "int"},
            {"column_name": "device", "type": "text"},
            {"column_name": "monthly_usage_min", "type": "int"},
            {"column_name": "name", "type": "text"},
            {"column_name": "support_case_ids", "type": "list<uuid>"},
        ]

        # Configure the session mock
        mock_session = MagicMock()
        mock_session.execute.side_effect = lambda cql, params=None: (
            mock_tables
            if "table_name" in cql
            else mock_columns_customer_support_transcripts
            if "customer_support_transcripts" in cql
            else mock_columns_family_plan_info
        )
        mock_get_session.return_value = mock_session

        # Create instance of DataAccess
        data_access = DataAccess()

        # Expected results
        expected_schemas = [
            "Table: customer_support_transcripts\nphone_number text\ntranscript_id uuid\ncustomer_name text\ninteraction_date timestamp\nissue_type text\nresolution_status text\ntranscript text",
            "Table: family_plan_info\nphone_number text\nfamily_member_phone_number text\nage int\ndevice text\nmonthly_usage_min int\nname text\nsupport_case_ids list<uuid>",
        ]

        # Test
        result = data_access.get_table_schemas(keyspace)
        self.assertEqual(result, expected_schemas)


if __name__ == "__main__":
    unittest.main()
