from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

import pandas as pd
from pathlib import Path
import joblib


class PredictPiece(BasePiece):

    def piece_function(self, input_data: InputModel) -> OutputModel:

        self.logger.info("PredictPiece started")
        self.logger.info(f"Model path: {input_data.model_path}")
        self.logger.info(f"Data path: {input_data.data_path}")

        model_path = Path(input_data.model_path)
        data_path = Path(input_data.data_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Prediction data not found: {data_path}")

        # ---- LOAD MODEL ----
        model = joblib.load(model_path)

        # ---- LOAD DATA ----
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        # ---- PREP FEATURES ----
        features = [c for c in df.columns if c != "datetime"]
        X = df[features]

        # ---- PREDICT ----
        predictions = model.predict(X)
        df["predicted_load"] = predictions

        # ---- SAVE OUTPUT ----
        output_path = Path(self.results_path) / "prediction_15min.parquet"
        df.to_parquet(output_path, index=False)

        # ---- DOMINO UI OUTPUT ----
        self.display_result = {
            "file_type": "parquet",
            "file_path": str(output_path)
        }

        message = "Prediction finished successfully"
        self.logger.info(message)

        return OutputModel(
            message=message,
            prediction_file_path=str(output_path)
        )
