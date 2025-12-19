from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    Input model for Fetch Energy Data Piece
    """

    load_csv: str = Field(
        title="Load CSV file path",
        description="Path to CSV file containing load data with datetime column"
    )

    production_csv: str = Field(
        title="Production CSV file path",
        description="Path to CSV file containing production data with datetime column"
    )

    prices_csv: str = Field(
        title="Prices CSV file path",
        description="Path to CSV file containing price data with datetime column"
    )


class OutputModel(BaseModel):
    """
    Output model for Fetch Energy Data Piece
    """

    message: str = Field(
        description="Execution result message"
    )

    output_path: str = Field(
        description="Path to merged Parquet file"
    )
