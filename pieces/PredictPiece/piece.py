from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime


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

        # ---- LOAD MODEL ----
        model = joblib.load(model_path)

        # ---- LOAD DATA ----
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        # ---- FEATURES (MUST MATCH TRAINING) ----
        target = "load_mw"
        features = [c for c in df.columns if c not in ["datetime", target]]
        X = df[features]

        # ---- PREDICT ----
        predictions = model.predict(X)

        df_out = df.copy()
        df_out["prediction_load_mw"] = predictions

        # ---- SAVE OUTPUT ----
        output_path = Path(self.results_path) / "predictions_15min.parquet"
        df_out.to_parquet(output_path, index=False)

        log_path = Path(self.results_path) / "prediction_log.txt"
        with open(log_path, "w") as f:
            f.write(f"Prediction time (UTC): {datetime.utcnow()}\n")
            f.write(f"Rows: {len(df_out)}\n")
            f.write(f"Features used: {features}\n")
            f.write(f"Model: {model_path.name}\n")

        message = "Prediction finished successfully"

        print(f"[SUCCESS] {message}")
        print(f"[SUCCESS] Predictions saved to {output_path}")

        # ✅ TOTO JE CELÁ OPRAVA
        return OutputModel(
            message=message,
            prediction_file_path=str(output_path),
            prediction_log_path=str(log_path)
        )
