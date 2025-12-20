from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import pandas as pd
import numpy as np


class PreprocessEnergyDataPiece(BasePiece):

    def piece_function(self, input_data: InputModel):

        print("[INFO] PreprocessEnergyDataPiece started")

        input_path = Path(input_data.input_path)
        train_out = Path(input_data.train_output_path)
        predict_out = Path(input_data.predict_output_path)

        if not input_path.exists():
            message = f"Input file not found: {input_path}"
            print(f"[ERROR] {message}")
            return OutputModel(message=message, train_file_path="", predict_file_path="")

        print(f"[INFO] Loading input data: {input_path}")

        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path, parse_dates=["datetime"])
        else:
            df = pd.read_parquet(input_path)

        print(f"[INFO] Rows loaded: {len(df)}")

        # BASIC CLEAN
        print("[INFO] Basic cleaning")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates(subset=["datetime"])
        df = df.sort_values("datetime")

        df = df.set_index("datetime")
        df = df.resample("1min").mean().ffill()

        print("[INFO] Creating 15-min training data")

        train_df = df.resample("15min").mean().reset_index()
        train_df["hour"] = train_df["datetime"].dt.hour
        train_df["day_of_week"] = train_df["datetime"].dt.dayofweek
        train_df["month"] = train_df["datetime"].dt.month

        if "load_kw" in train_df.columns:
            train_df["lag_24h"] = train_df["load_kw"].shift(96)
            train_df["rolling_4h"] = train_df["load_kw"].rolling(16, min_periods=1).mean()

        train_df["sin_hour"] = np.sin(2 * np.pi * train_df["hour"] / 24)
        train_df["cos_hour"] = np.cos(2 * np.pi * train_df["hour"] / 24)
        train_df = train_df.dropna()

        print("[INFO] Creating daily prediction data")

        predict_df = df.resample("1D").mean().reset_index()

        train_out.parent.mkdir(parents=True, exist_ok=True)
        predict_out.parent.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(train_out, index=False)
        predict_df.to_parquet(predict_out, index=False)

        print("[SUCCESS] Preprocessing finished")

        self.display_result = {
            "train_file": str(train_out),
            "predict_file": str(predict_out)
        }

        return OutputModel(
            message="Preprocessing completed successfully",
            train_file_path=str(train_out),
            predict_file_path=str(predict_out)
        )
