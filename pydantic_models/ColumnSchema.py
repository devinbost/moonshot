from pydantic import BaseModel


class ColumnSchema(BaseModel):
    column_name: str
    type: str
    clustering_order: str
    kind: str
    position: int
