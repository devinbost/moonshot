from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json

from pydantic_models.ColumnSchema import ColumnSchema


class TableSchema(BaseModel):
    table_name: str
    keyspace_name: str
    columns: Optional[List[ColumnSchema]] = None
    indexes: Optional[List[str]] = None
    keys: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
