
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import pandas as pd
import math


class PreprocessEnergyDataPiece(BasePiece):
    """
    Preprocess energy data + generate realistic future horizon
    Future is created by repeating last week pattern (not flat values)
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        print("[INFO] PreprocessEnergyDataPiece started")

        input_path = Path(input_data.input_path)
        forecast_hours = getattr(input_data, "forecast_hours", 24)

        print(f"[INFO] Using input file: {input_path}")
        print(f"[INFO] Forecast horizon: {forecast_hours} hours")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # ---- LOAD ----
        df = pd.read_parquet(input_path)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates(subset=["datetime"])
        df = df.sort_values("datetime")
        df = df.set_index("datetime")

        # ---- RESAMPLE TO 15MIN (canonical resolution) ----
        df_15min = df.resample("15min").mean().ffill()

        # training dataset (keep same resolution as reality)
        train_df = df_15min.copy()

        # ---- FUTURE GENERATION USING LAST WEEK PATTERN ----
        print("[INFO] Building future dataset using last week replay")

        last_timestamp = df_15min.index.max()

        # last 7 days pattern
        last_week = df_15min.last("7D")

        if len(last_week) == 0:
            raise ValueError("Not enough historical data to build last-week pattern")

        # how many steps needed
        steps = int(forecast_hours * 60 / 15)
        print(f"[INFO] Need future steps: {steps}")

        repeat_count = math.ceil(steps / len(last_week))
        future_pattern = pd.concat([last_week] * repeat_count)

        future_pattern = future_pattern.iloc[:steps].copy()

        # shift timestamps forward
        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=15),
            periods=steps,
            freq="15min"
        )

        future_pattern.index = future_index

        # combine history + future
        predict_df = pd.concat([df_15min, future_pattern])

        # ---- SAVE ----
        train_path = Path(self.results_path) / "train_dataset.parquet"
        predict_path = Path(self.results_path) / "predict_dataset_15min.parquet"

        train_df.reset_index().to_parquet(train_path, index=False)
        predict_df.reset_index().to_parquet(predict_path, index=False)

        print("[SUCCESS] Preprocessing finished")
        print(f"[INFO] Train rows: {len(train_df)}")
        print(f"[INFO] Predict rows: {len(predict_df)}")

        self.display_result = {
            "file_type": "parquet",
            "file_path": str(predict_path)
        }

        return OutputModel(
            message=f"Preprocessing finished (future from last week, +{forecast_hours}h)",
            train_file_path=str(train_path),
            predict_file_path=str(predict_path)
        )
