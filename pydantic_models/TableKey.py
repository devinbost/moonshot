from pydantic import BaseModel


class TableKey(BaseModel):
    column_name: str
    clustering_order: str
    kind: str
    position: int
