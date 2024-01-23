from pydantic import BaseModel
from typing import Optional


class Person(BaseModel):
    id: str
    full_name: str
    url: str
    lifespan: Optional[str]

    def get_birth_date(self):
        lifespan_parts = self.lifespan.split("-")
        # Need to make this more robust in case it's empty or doesn't have a -
        birth_date = lifespan_parts[0]
        return birth_date

    def get_death_date(self):
        lifespan_parts = self.lifespan.split("-")
        # Need to make this more robust in case it's empty or doesn't have a -
        death_date = lifespan_parts[1]
        return death_date
