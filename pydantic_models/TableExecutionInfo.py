from typing import List, Optional

from cassandra.cluster import ResultSet
from cassandra.cqltypes import UUID, Decimal
from cassandra.query import datetime, ValueSequence
from cassandra.util import OrderedMap
from pydantic import BaseModel
import json
from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.ExecutionFailure import ExecutionFailure
from pydantic_models.TableSchema import TableSchema


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, UUID):
            return str(o)
        elif isinstance(o, Decimal):
            return float(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, OrderedMap):
            return dict(o)
        elif isinstance(o, ValueSequence):
            return list(o)
        return json.JSONEncoder.default(self, o)


class TableExecutionInfo(BaseModel):
    """
    Represents the execution information of a table, including its schema,
    execution counter, rows processed, and any prior failures.

    Attributes:
        table_schema (TableSchema): The schema of the table.
        execution_counter (str): A string identifier for the number of times the table has been executed.
        rows (Optional[Row]): The rows processed during the execution. The type 'Row' should be defined elsewhere.
        prior_failures (Optional[List[ExecutionFailure]]): A list of prior execution failures, if any.
    """

    table_schema: TableSchema
    execution_counter: str
    rows: Optional[List[dict]]
    prior_failures: Optional[List[ExecutionFailure]]

    def to_lcel_json_prefixed(self) -> str:
        """
        Converts the instance data into a JSON string with a specific prefix format.
        This method also omits any empty properties from the JSON output.

        Returns:
            str: A JSON string representation of the instance with a prefixed format.
        """
        # Do something like this in the future:
        # for d in dict_list:
        #     result_string += ",".join(map(str, d.values())) + "\n"
        json_str = json.dumps(
            self.dict(), cls=CustomJSONEncoder
        )  # Convert instance to a dictionary, then to a JSON string.
        json_str = f"""
    TABLE EXECUTION INFO:

{json_str}"""
        final = json_str.replace("{", "{{").replace("}", "}}")
        return final

    def to_lcel_json_prefixed_with_only_last_failure(self) -> str:
        """
        Generates a JSON string representation of the instance including only the last failure information,
        if available, along with the table schema.

        Returns:
            str: A JSON string representation of the table schema and the last failure information.
        """
        last_failure = self.prior_failures[-1] if self.prior_failures else None
        out = (
            self.table_schema.to_lcel_json_prefixed()
            + "\n"
            + (last_failure.to_lcel_json_prefixed() if last_failure else "")
        )
        json_str = """\n\nTABLE EXECUTION INFO WITH LAST FAILURE:\n\n""" + out
        return json_str
