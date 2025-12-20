from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import pandas as pd
import xgboost as xgb
import joblib


class TrainModelPiece(BasePiece):
    """
    Train XGBoost model on preprocessed energy data
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        print("[INFO] TrainModelPiece started")

        train_path = Path(input_data.train_file_path)
        print(f"[INFO] Using training file: {train_path}")

        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")

        # ---- LOAD DATA ----
        df = pd.read_parquet(train_path)

        if "datetime" in df.columns:
            df = df.drop(columns=["datetime"])

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        print(f"[INFO] Rows: {len(df)}")
        print(f"[INFO] Features: {list(X.columns)}")

        # ---- TRAIN MODEL ----
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="reg:squarederror",
            n_jobs=1
        )

        model.fit(X, y)

        # ---- SAVE MODEL ----
        model_path = Path(self.results_path) / "xgboost_energy_model.joblib"
        joblib.dump(model, model_path)

        print(f"[SUCCESS] Model saved to {model_path}")

        # POVINNÉ PRE DOMINO
        self.display_result = {
            "file_type": "model",
            "file_path": str(model_path)
        }

        return OutputModel(
            message="Model trained successfully",
            model_path=str(model_path)
        )
