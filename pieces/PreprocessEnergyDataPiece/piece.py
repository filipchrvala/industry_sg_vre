from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
import pandas as pd


class PreprocessEnergyDataPiece(BasePiece):
    """
    Basic cleaning + resampling of merged energy data
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        print("[INFO] PreprocessEnergyDataPiece started")

        input_path = Path(input_data.input_path)
        print(f"[INFO] Using input file: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_parquet(input_path)

        # ---------- BASIC CLEAN ----------
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates(subset=["datetime"])
        df = df.sort_values("datetime")
        df = df.set_index("datetime")

        # ---------- RESAMPLE ----------
        df_1min = df.resample("1min").mean().ffill()
        df_15min = df.resample("15min").mean().ffill()

        # ---------- SAVE ----------
        train_path = Path(self.results_path) / "train_dataset_1min.parquet"
        forecast_path = Path(self.results_path) / "forecast_dataset_15min.parquet"

        df_1min.reset_index().to_parquet(train_path, index=False)
        df_15min.reset_index().to_parquet(forecast_path, index=False)

        print("[SUCCESS] Preprocessing finished")

        # ✅ POVINNÉ PRE DOMINO UI
        self.display_result = {
            "file_type": "parquet",
            "file_path": str(train_path)
        }

        return OutputModel(
            message="Preprocessing finished successfully",
            train_data_path=str(train_path),
            forecast_data_path=str(forecast_path)
        )
