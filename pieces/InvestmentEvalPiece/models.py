from pydantic import BaseModel, Field


class InputModel(BaseModel):
    kpi_results_csv: str = Field(
        description="Path to kpi_results.csv from KPIPiece"
    )

    battery_summary_csv: str = Field(
        description="Path to battery summary csv (cycles etc.)",
        default=""
    )

    investment_config_yml: str = Field(
        description="Path to investment config yaml"
    )


class OutputModel(BaseModel):
    message: str
    investment_evaluation_json: str
