from pydantic import BaseModel


class ColumnSchema(BaseModel):
    column_name: str
    column_type: str
