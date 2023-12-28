from typing import Any

from pydantic import BaseModel
import json


class PropertyInfo(BaseModel):
    """
    A Pydantic model representing a property with its name, type, and value.

    Attributes:
        property_name (str): The name of the property.
        property_type (str): The type of the property, represented as a string.
        property_value (Any): The value of the property. Can be of any type.
    """

    property_name: str
    property_type: str
    property_value: Any

    def to_string_human(self):
        """
        Returns a string representation of the property.

        Returns:
            str: A formatted string displaying the property's details, designed for use by an LCEL prompt.
        """
        output = f"""
    Property:
        name: {self.property_name}
        type: {self.property_type}
        value (as str): {str(self.property_value)}
"""
        return output

    def to_lcel_json_prefixed(self) -> str:
        json_str = "\nProperty: \n"
        json_str += json.dumps(self.dict())
        json_str = json_str.replace("{", "{{").replace("}", "}}")
        return json_str
