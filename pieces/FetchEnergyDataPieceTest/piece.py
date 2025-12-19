from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from pathlib import Path


class FetchEnergyDataPieceTest(BasePiece):

    def piece_function(self, input_data: InputModel):

        print("[INFO] FetchEnergyDataPieceTest started")

        load_csv = Path(input_data.load_csv)
        production_csv = Path(input_data.production_csv)
        prices_csv = Path(input_data.prices_csv)

        for f in [load_csv, production_csv, prices_csv]:
            if not f.exists():
                message = f"Input file not found: {f}"
                print(f"[ERROR] {message}")
                return OutputModel(message=message, file_path="")

        print("[INFO] Reading CSV files")

        load_df = pd.read_csv(load_csv, parse_dates=["datetime"])
        production_df = pd.read_csv(production_csv, parse_dates=["datetime"])
        prices_df = pd.read_csv(prices_csv, parse_dates=["datetime"])

        print("[INFO] Merging data")

        load_df = load_df.set_index("datetime")
        production_df = production_df.set_index("datetime")
        prices_df = prices_df.set_index("datetime")

        merged_df = (
            load_df
            .join(production_df, how="outer")
            .join(prices_df, how="outer")
            .reset_index()
        )

        if "production_ton" in merged_df.columns:
            merged_df["production_ton"] = merged_df["production_ton"].ffill()

        if "price_eur_mwh" in merged_df.columns:
            merged_df["price_eur_mwh"] = merged_df["price_eur_mwh"].ffill()

        output_file = Path(input_data.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(output_file, index=False)

        message = f"Energy data merged successfully ({len(merged_df)} rows)"
        print(f"[SUCCESS] {message}")

        self.display_result = {
            "file_type": "parquet",
            "file_path": str(output_file)
        }

        return OutputModel(message=message, file_path=str(output_file))
