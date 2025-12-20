from pydantic import BaseModel, Field


class InputModel(BaseModel):
    train_file_path: str = Field(
        description="Path to training features parquet file"
    )


class OutputModel(BaseModel):
    message: str
    model_path: str