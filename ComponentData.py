from typing import Optional

from pydantic import BaseModel


class ComponentData(BaseModel):
    id: str
    library: str
    class_name: str
    access_type: str
    component_name: str
    access_type: str  # e.g. "method", "constructor", "property"
    component_type: Optional[
        str
    ]  # Needs to be changed to required after we get things working.
    params: dict[
        str, str
    ] = {}  # key/value pairs of named parameters (names and their values)
    output_var: str
