from typing import List
from pydantic import BaseModel
from pydantic_models.PropertyInfo import PropertyInfo
import json


class UserInfo(BaseModel):
    properties: List[PropertyInfo]

    def to_lcel_json(self) -> str:
        json_str = json.dumps(self.dict())
        return json_str.replace("{", "{{").replace("}", "}}")

    def to_lcel_json_prefixed(self) -> str:
        json_str = """
--------
USER INFO: \n"""
        for prop in self.properties:
            json_str += prop.to_lcel_json_prefixed()
        return json_str

    def to_json(self) -> str:
        json_str = json.dumps(self.dict())
        return json_str

    def to_string_human(self) -> str:
        """
        Returns a string representation of user information.

        Returns:
            str: A formatted string displaying each property's details, designed for use by an LCEL prompt.
        """
        output = "\nUser Info:\n"
        for prop in self.properties:
            output += prop.to_string_human()
        return output
