from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
import numpy as np
from pathlib import Path


class PreprocessSolargisPiece(BasePiece):
    
    def piece_function(self, input_data: InputModel):

        print(f"[INFO] Preprocessing {input_data.input_path}")
    
        # Load data
        df = pd.read_csv(input_data.input_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, dayfirst=True)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M')
    
        # Filter out night hours
        original_rows = len(df)
        df = df[df['GHI'] > 1].copy()
    
        # Feature engineering for CIS panels
        df['diffuse_fraction'] = np.where(df['GHI'] > 0, df['DIF'] / df['GHI'], 0)
        df['solar_elevation_sin'] = np.sin(np.radians(df['SE']))
        df['hour_of_day'] = df['datetime'].dt.hour
    
        # Save processed data
        file_path = str(Path(self.results_path) / "processed_solargis.csv") 
        df.to_csv(file_path, index=False)
        processed_rows = len(df)
    
        message = f"[SUCCESS] Preprocessed data saved to {file_path}"
        print(message)
        
        # Set display result
        self.display_result = {
            "file_type": "csv",
            "file_path": file_path if processed_rows > 0 else ""
        }
    
        return OutputModel(
            message = message,
            processed_rows = processed_rows,
            file_path = file_path if processed_rows > 0 else ""
        )
