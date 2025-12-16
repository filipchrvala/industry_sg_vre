from pydantic import BaseModel, Field

class InputModel(BaseModel):
    """
    Fetch Solargis Data Piece Input Model
    """
    input_path: str = Field(
        title="Path to input data files",
        default='/home/shared_storage/input_data/InputSolargisFile.csv',
        description="The path to input meteo data files (supports wildcards: *.docx, *.csv, or directory paths)"
    )

    input_filetype: str = Field(
        title="Input file type",
        default='csv',
        description="The type of input file: 'docx' for Word documents or 'csv' for CSV files"
    )    
    
class OutputModel(BaseModel):
    """
    Fetch Solargis Data Piece Output Model
    """
    message: str = Field(
        default="",
        description="Output message to log"
    )

    file_path: str = Field(
        description="The path to the output CSV file"
    )
