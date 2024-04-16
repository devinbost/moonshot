from typing import Optional

from pydantic import BaseModel

class KnowledgeTuple(BaseModel):
    head: str
    head_description: str
    relation: str
    tail: str
    tail_description: str

    def get_as_tagged(self):
        output = f"<head_entity>{self.head}</head_entity><head_entity_description>{self.head_description}</head_entity_description><relation>{self.relation}</relation><tail_entity>{self.tail}</tail_entity><tail_entity_description>{self.tail_description}</tail_entity_description>"
        return output