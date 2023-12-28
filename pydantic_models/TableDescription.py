from pydantic import BaseModel


class TableDescription(BaseModel):
    table_name: str
    column_name: str
    description: str
