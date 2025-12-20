from pydantic import BaseModel, Field


class InputModel(BaseModel):
    train_file_path: str = Field(
        description="Path to training dataset parquet (1min resolution)"
    )


class OutputModel(BaseModel):
    message: str
    model_path: str
