from typing import List, Dict, Any
from pydantic import BaseModel
import json
from pydantic_models.ColumnSchema import ColumnSchema


class TableSchema(BaseModel):
    table_name: str
    schema_name: str
    columns: List[ColumnSchema]

    def to_lcel_json(self) -> str:
        json_str = json.dumps(self.dict())
        return json_str.replace("{", "{{").replace("}", "}}")

    def to_lcel_json_prefixed(self) -> str:
        json_str = json.dumps(self.dict())
        json_str = f"""
TABLE SCHEMA:

{json_str}"""
        return json_str.replace("{", "{{").replace("}", "}}")

    def to_json(self) -> str:
        json_str = json.dumps(self.dict())
        return json_str
