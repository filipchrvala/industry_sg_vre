from pydantic import BaseModel, Field


class InputModel(BaseModel):
    load_csv: str = Field(default="/home/shared_storage/load.csv")
    production_csv: str = Field(default="/home/shared_storage/production.csv")
    prices_csv: str = Field(default="/home/shared_storage/prices.csv")
    output_path: str = Field(default="/home/shared_storage/data/merged_energy_data.parquet")


class OutputModel(BaseModel):
    message: str = Field(default="")
    file_path: str
