import unittest

from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.PropertyInfo import PropertyInfo
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo


class TestTableSchemaToJson(unittest.TestCase):
    def test_to_lcel_json(self):
        table_schema = TableSchema(
            table_name="test_table",
            schema_name="test_schema",
            columns=[
                ColumnSchema(column_name="id", column_type="int"),
                ColumnSchema(column_name="name", column_type="text"),
            ],
        )

        json_output = table_schema.to_lcel_json()
        expected_output = (
            '{{"table_name": "test_table", "schema_name": "test_schema", "columns": [{{"column_name": '
            '"id", "column_type": "int"}}, {{"column_name": "name", "column_type": "text"}}]}}'
        )  # Double curly braces are to ensure LCEL doesn't misinterpret it.

        self.assertEqual(json_output, expected_output)

    def test_to_json(self):
        table_schema = TableSchema(
            table_name="test_table",
            schema_name="test_schema",
            columns=[
                ColumnSchema(column_name="id", column_type="int"),
                ColumnSchema(column_name="name", column_type="text"),
            ],
        )

        json_output = table_schema.to_json()
        expected_output = (
            '{"table_name": "test_table", "schema_name": "test_schema", "columns": [{"column_name": '
            '"id", "column_type": "int"}, {"column_name": "name", "column_type": "text"}]}'
        )

        self.assertEqual(json_output, expected_output)


class TestPropertyInfo(unittest.TestCase):
    def test_to_string(self):
        prop_info = PropertyInfo(
            property_name="ExampleProperty",
            property_type="string",
            property_value="ExampleValue",
        )
        expected = f"""
    Property:
        name: ExampleProperty
        type: string
        value (as str): ExampleValue
"""
        actual = prop_info.to_string_human()
        self.assertEqual(expected, actual)

    def test_to_lcel_json_prefixed(self):
        # Create an instance of PropertyInfo
        prop_info = PropertyInfo(
            property_name="ExampleProperty",
            property_type="string",
            property_value="ExampleValue",
        )

        # Expected JSON string
        expected_json = '\nProperty: \n{{"property_name": "ExampleProperty", "property_type": "string", "property_value": "ExampleValue"}}'

        # Test the to_lcel_json_prefixed method
        self.assertEqual(expected_json, prop_info.to_lcel_json_prefixed())


class TestUserInfoToJson(unittest.TestCase):
    def test_to_json(self):
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="name",
                    property_type="text",
                    property_value="John Doe",
                ),
            ]
        )

        json_output = user_info.to_lcel_json()
        expected_output = (
            '{{"properties": [{{"property_name": "age", "property_type": "int", "property_value": 30}}, '
            '{{"property_name": "name", "property_type": "text", "property_value": "John Doe"}}]}}'
        )
        # expected_output = expected_output.replace("{", "{{").replace("}", "}}")

        self.assertEqual(json_output, expected_output)

    def test_to_string(self):
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="Age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="Name",
                    property_type="string",
                    property_value="John Doe",
                ),
            ]
        )
        # Expected output string
        expected_output = """
User Info:

    Property:
        name: Age
        type: int
        value (as str): 30
        
    Property:
        name: Name
        type: string
        value (as str): John Doe
"""

        # Test the to_string method
        self.assertEqual(expected_output, user_info.to_string_human())

    def test_to_lcel_json_prefixed(self):
        # Create UserInfo with a list of PropertyInfo
        user_info = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="Age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="Name",
                    property_type="string",
                    property_value="John Doe",
                ),
            ]
        )

        # Expected JSON string
        expected_json = """\n--------\nUSER INFO: \n\nProperty: \n{{"property_name": "Age", "property_type": "int", "property_value": 30}}\nProperty: \n{{"property_name": "Name", "property_type": "string", "property_value": "John Doe"}}"""

        # Test the to_lcel_json_prefixed method
        self.assertEqual(expected_json, user_info.to_lcel_json_prefixed())


class TestTableSchema(unittest.TestCase):
    def test_to_lcel_json_prefixed(self):
        # Create a TableSchema instance with a list of ColumnSchema
        table_schema = TableSchema(
            table_name="example_table",
            schema_name="example_schema",
            columns=[
                ColumnSchema(column_name="id", column_type="int"),
                ColumnSchema(column_name="name", column_type="string"),
            ],
        )

        # Expected JSON string
        expected_json = """
TABLE SCHEMA:

{{"table_name": "example_table", "schema_name": "example_schema", "columns": [{{"column_name": "id", "column_type": "int"}}, {{"column_name": "name", "column_type": "string"}}]}}"""

        # Test the to_lcel_json_prefixed method
        self.assertEqual(expected_json, table_schema.to_lcel_json_prefixed())


if __name__ == "__main__":
    unittest.main()
