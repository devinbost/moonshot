from pydantic import BaseModel
import json


class ExecutionFailure(BaseModel):
    failed_query: str
    error_message: str

    def to_lcel_json_prefixed(self) -> str:
        json_str = "\nExecutionFailure: \n"
        json_str += json.dumps(self.dict())
        json_str = json_str.replace("{", "{{").replace("}", "}}")
        return json_str
