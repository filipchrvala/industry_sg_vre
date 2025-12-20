from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

import pandas as pd
from pathlib import Path
import joblib

class PredictPiece(BasePiece):

    def piece_function(self, input_data: InputModel) -> OutputModel:
        print("[INFO] PredictPiece started")
        print(f"[INFO] Model path: {input_data.model_path}")
        print(f"[INFO] Data path: {input_data.data_path}")

        model_path = Path(input_data.model_path)
        data_path = Path(input_data.data_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Prediction data not found: {data_path}")

        model = joblib.load(model_path)

        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        features = df.drop(columns=["datetime"]) if "datetime" in df.columns else df
        preds = model.predict(features)

        out_df = df.copy()
        out_df["predicted_load"] = preds

        output_path = Path(self.results_path) / "prediction_15min.parquet"
        out_df.to_parquet(output_path, index=False)

        self.display_result = {
            "file_type": "parquet",
            "file_path": str(output_path)
        }

        return OutputModel(
            message="Prediction finished successfully",
            prediction_path=str(output_path)
        )
