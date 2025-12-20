from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_path: str = Field(
        default="/home/shared_storage/data/merged_energy_data.parquet",
        description="Path to raw merged CSV or Parquet file"
    )

    train_output_path: str = Field(
        default="/home/shared_storage/data/train_features_15min.parquet",
        description="Output Parquet for model training"
    )

    predict_output_path: str = Field(
        default="/home/shared_storage/data/predict_features_daily.parquet",
        description="Output Parquet for daily load forecast"
    )


class OutputModel(BaseModel):
    message: str = Field(default="")
    train_file_path: str
    predict_file_path: str
