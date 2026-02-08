
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime


class PredictPiece(BasePiece):
    """
    FINAL production-ready prediction piece for energy simulation pipeline.

    Output MUST match SimulatePiece expectations:
    datetime, load_kw, price_eur_mwh (+ optional features)

    load_kw column = MODEL FORECAST (not historical)
    """

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
        print("[INFO] Loading model")
        model = joblib.load(model_path)

        # ---- LOAD DATA ----
        print("[INFO] Loading prediction dataset")
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        if "datetime" not in df.columns:
            raise ValueError("Prediction dataset must contain 'datetime' column")

        # ---- FEATURE PREP ----
        target = "load_kw"

        # features = everything except datetime + target
        features = [c for c in df.columns if c not in ["datetime", target]]

        if len(features) == 0:
            raise ValueError("No feature columns found for prediction")

        print(f"[INFO] Features used: {features}")

        X = df[features]

        # ---- PREDICT ----
        print("[INFO] Running model prediction")
        predictions = model.predict(X)

        # ---- BUILD OUTPUT ----
        df_out = df.copy()

        # IMPORTANT: overwrite load_kw with prediction
        df_out[target] = predictions

        # ensure required columns exist for simulation
        required_cols = ["datetime", "load_kw"]

        for col in required_cols:
            if col not in df_out.columns:
                raise ValueError(f"Missing required column in output: {col}")

        # ---- SAVE CSV FOR SIMULATION ----
        output_path = Path(self.results_path) / "predictions_15min.csv"
        df_out.to_csv(output_path, index=False)

        # ---- LOG ----
        log_path = Path(self.results_path) / "prediction_log.txt"
        with open(log_path, "w") as f:
            f.write(f"Prediction time (UTC): {datetime.utcnow()}\n")
            f.write(f"Rows: {len(df_out)}\n")
            f.write(f"Forecast horizon included: YES (+future rows if present)\n")
            f.write(f"Features used: {features}\n")
            f.write(f"Model: {model_path.name}\n")

        message = "Forecast generated successfully for simulation"

        print(f"[SUCCESS] {message}")
        print(f"[SUCCESS] Output saved to {output_path}")

        return OutputModel(
            message=message,
            prediction_file_path=str(output_path)
        )
