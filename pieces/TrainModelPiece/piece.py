from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBRegressor
from datetime import datetime


class TrainModelPiece(BasePiece):

    def piece_function(self, input_data: InputModel) -> OutputModel:

        print(f"[INFO] TrainModelPiece started")
        print(f"[INFO] Using training data: {input_data.data_path}")

        data_path = Path(input_data.data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        # ---- LOAD DATA ----
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        # ---- FEATURES / TARGET ----
        target = "load_mw"

        # ---- SAFETY CHECK ----
        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        features = [c for c in df.columns if c not in ["datetime", target]]

        X = df[features]
        y = df[target]

        # ---- TRAIN MODEL ----
        model = XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.05,
            max_depth=5,
            n_estimators=300
        )

        model.fit(X, y)

        # ---- SAVE OUTPUTS ----
        model_path = Path(self.results_path) / "xgboost_model.pkl"
        log_path = Path(self.results_path) / "training_log.txt"

        joblib.dump(model, model_path)

        with open(log_path, "w") as f:
            f.write(f"Model trained at {datetime.utcnow()}\n")
            f.write(f"Rows: {len(df)}\n")
            f.write(f"Features: {features}\n")

        message = "Model trained successfully"

        print(f"[SUCCESS] {message}")
        print(f"[SUCCESS] Model saved to {model_path}")

        return OutputModel(
            message=message,
            model_file_path=str(model_path),
            train_log_path=str(log_path)
        )
