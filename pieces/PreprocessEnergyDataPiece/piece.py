from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import pandas as pd
import numpy as np


class PreprocessEnergyDataPiece(BasePiece):

    def piece_function(self, input_data: InputModel):

        print("[INFO] PreprocessEnergyDataPiece started")
        print(f"[INFO] Using input file: {input_data.input_path}")

        input_path = Path(input_data.input_path)

        if not input_path.exists():
            msg = f"Input file not found: {input_path}"
            print(f"[ERROR] {msg}")
            return OutputModel(msg, "", "")

        df = pd.read_parquet(input_path)

        # --- BASIC CLEAN ---
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates(subset=["datetime"])
        df = df.sort_values("datetime")
        df = df.set_index("datetime")

        # unify to 1-minute grid
        df = df.resample("1min").mean().ffill()

        # --- TRAIN DATA (15 min) ---
        train_df = df.resample("15min").mean().reset_index()
        train_df["hour"] = train_df["datetime"].dt.hour
        train_df["day_of_week"] = train_df["datetime"].dt.dayofweek
        train_df["month"] = train_df["datetime"].dt.month

        if "load_kw" in train_df.columns:
            train_df["yesterday_load_kw"] = train_df["load_kw"].shift(96)
            train_df["rolling_4h"] = train_df["load_kw"].rolling(16).mean()

        train_df["sin_hour"] = np.sin(2 * np.pi * train_df["hour"] / 24)
        train_df["cos_hour"] = np.cos(2 * np.pi * train_df["hour"] / 24)
        train_df = train_df.dropna()

        # --- PREDICT DATA (daily) ---
        predict_df = df.resample("1D").mean().reset_index()

        train_out = Path(self.results_path) / "train_features_15min.parquet"
        predict_out = Path(self.results_path) / "predict_features_daily.parquet"

        train_df.to_parquet(train_out, index=False)
        predict_df.to_parquet(predict_out, index=False)

        print("[SUCCESS] Preprocessing finished")

        self.display_result = {
            "train_file": str(train_out),
            "predict_file": str(predict_out)
        }

        return OutputModel(
            "Preprocessing completed",
            str(train_out),
            str(predict_out)
        )
