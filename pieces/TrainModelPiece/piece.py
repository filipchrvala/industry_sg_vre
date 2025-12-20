from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import pandas as pd
import xgboost as xgb


class TrainModelPiece(BasePiece):

    def piece_function(self, input_data: InputModel):

        print("[INFO] TrainModelPiece started")
        print(f"[INFO] Using training data: {input_data.train_file_path}")

        data_path = Path(input_data.train_file_path)

        if not data_path.exists():
            return OutputModel(
                message=f"Training file not found: {data_path}",
                model_path=""
            )

        df = pd.read_parquet(data_path)

        if "load_kw" not in df.columns:
            return OutputModel(
                message="Column 'load_kw' not found in training data",
                model_path=""
            )

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

        model.fit(X, y)

        model_path = Path(self.results_path) / "xgboost_energy_model.json"
        model.save_model(model_path)

        print("[SUCCESS] Model training completed")

        self.display_result = {
            "model_path": str(model_path)
        }

        return OutputModel(
            message="Model trained successfully",
            model_path=str(model_path)
        )