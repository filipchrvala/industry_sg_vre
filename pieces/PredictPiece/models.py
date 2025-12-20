from pydantic import BaseModel

class InputModel(BaseModel):
    model_path: str
    data_path: str

class OutputModel(BaseModel):
    message: str
    prediction_path: str
