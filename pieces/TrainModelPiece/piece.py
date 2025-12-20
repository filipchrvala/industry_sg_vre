from domino.base_piece import BasePiece
from pathlib import Path
import pandas as pd
import xgboost as xgb


class TrainModelPiece(BasePiece):

    def piece_function(self, train_file_path: str):
        print("[INFO] TrainModelPiece started")
        print(f"[INFO] Training file: {train_file_path}")

        train_path = Path(train_file_path)

        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")

        df = pd.read_parquet(train_path)

        if "load_kw" not in df.columns:
            raise ValueError("Column 'load_kw' not found in training data")

        X = df.drop(columns=["load_kw", "datetime"], errors="ignore")
        y = df["load_kw"]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        print("[INFO] Training XGBoost model")
        model.fit(X, y)

        model_path = Path(self.results_path) / "xgboost_energy_model.json"
        model.save_model(model_path)

        print("[SUCCESS] Model trained successfully")

        # POVINNÉ pre Domino
        self.display_result = {
            "file_type": "model",
            "file_path": str(model_path)
        }

        return {
            "message": "Model trained successfully",
            "model_path": str(model_path)
        }
