from pydantic import BaseModel


class TableIndex(BaseModel):
    column_name: str
