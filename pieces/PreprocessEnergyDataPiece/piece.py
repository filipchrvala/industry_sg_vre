
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import pandas as pd


class PreprocessEnergyDataPiece(BasePiece):
    """
    Preprocess + configurable forecast horizon from UI (hours ahead)
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        print("[INFO] PreprocessEnergyDataPiece started")

        input_path = Path(input_data.input_path)
        forecast_hours = getattr(input_data, "forecast_hours", 24)  # default 24h

        print(f"[INFO] Using input file: {input_path}")
        print(f"[INFO] Forecast horizon from UI: {forecast_hours} hours")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_parquet(input_path)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates(subset=["datetime"])
        df = df.sort_values("datetime")
        df = df.set_index("datetime")

        # training dataset
        df_1min = df.resample("1min").mean().ffill()

        # prediction dataset base
        df_15min = df.resample("15min").mean().ffill()

        # -------- FUTURE HORIZON FROM UI --------
        steps = int(forecast_hours * 60 / 15)

        print(f"[INFO] Generating future horizon: {steps} rows (15min resolution)")

        last_time = df_15min.index.max()

        future_index = pd.date_range(
            start=last_time + pd.Timedelta(minutes=15),
            periods=steps,
            freq="15min"
        )

        future_df = pd.DataFrame(index=future_index)

        # baseline future features = last known value
        for col in df_15min.columns:
            future_df[col] = df_15min[col].iloc[-1]

        predict_df = pd.concat([df_15min, future_df])

        # -------- SAVE --------
        train_path = Path(self.results_path) / "train_dataset_1min.parquet"
        predict_path = Path(self.results_path) / "predict_dataset_15min.parquet"

        df_1min.reset_index().to_parquet(train_path, index=False)
        predict_df.reset_index().to_parquet(predict_path, index=False)

        print("[SUCCESS] Preprocessing finished")
        print(f"[INFO] Predict dataset rows: {len(predict_df)}")

        self.display_result = {
            "file_type": "parquet",
            "file_path": str(train_path)
        }

        return OutputModel(
            message=f"Preprocessing finished (forecast +{forecast_hours}h)",
            train_file_path=str(train_path),
            predict_file_path=str(predict_path)
        )
