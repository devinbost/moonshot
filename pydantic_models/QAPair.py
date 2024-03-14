from pydantic import BaseModel
class QAPair(BaseModel):
    question: str
    answer: str