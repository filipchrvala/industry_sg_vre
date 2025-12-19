print(">>> FetchEnergyPiece module imported <<<")

from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from pathlib import Path


class FetchEnergyPiece(BasePiece):
    """
    Domino piece responsible for:
    - loading 3 CSV files (load, production, prices)
    - merging them on datetime column
    - storing merged output as Parquet
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        self.logger.info("=== FetchEnergyPiece started ===")

        try:
            # Log input parameters
            self.logger.info(
                "Input paths received",
                extra={
                    "load_csv": input_data.load_csv,
                    "production_csv": input_data.production_csv,
                    "prices_csv": input_data.prices_csv,
                }
            )

            # Convert input paths to Path objects
            load_csv = Path(input_data.load_csv)
            production_csv = Path(input_data.production_csv)
            prices_csv = Path(input_data.prices_csv)

            # Define output path
            output_parquet = Path(self.results_path) / "merged_energy_data.parquet"
            self.logger.info(f"Output will be written to: {output_parquet}")

            # Validate input files
            for file_path in [load_csv, production_csv, prices_csv]:
                self.logger.info(f"Checking existence of file: {file_path}")
                if not file_path.exists():
                    message = f"Input file not found: {file_path}"
                    self.logger.error(message)
                    raise FileNotFoundError(message)

            # Read CSV files
            self.logger.info("Reading input CSV files")
            load_df = self._read_csv(load_csv, "load")
            production_df = self._read_csv(production_csv, "production")
            prices_df = self._read_csv(prices_csv, "prices")

            # Merge data
            self.logger.info("Merging data frames")
            merged_df = self._merge_data(load_df, production_df, prices_df)

            self.logger.info(
                "Merge completed",
                extra={
                    "rows": len(merged_df),
                    "columns": list(merged_df.columns),
                }
            )

            # Save output
            self.logger.info("Saving merged data to Parquet")
            merged_df.to_parquet(output_parquet, index=False)

            self.logger.info("Parquet file saved successfully")

            # Display result for Domino UI
            self.display_result = {
                "file_type": "parquet",
                "file_path": str(output_parquet),
            }

            self.logger.info("=== FetchEnergyPiece finished successfully ===")

            return OutputModel(
                message=f"Data merged successfully ({len(merged_df)} rows)",
                output_path=str(output_parquet),
            )

        except Exception as exc:
            # Centralized error logging
            self.logger.exception("FetchEnergyPiece failed with an exception")
            return OutputModel(
                message=str(exc),
                output_path="",
            )

    def _read_csv(self, file_path: Path, label: str) -> pd.DataFrame:
        """
        Read CSV file and parse datetime column.
        """
        self.logger.info(f"Reading {label} CSV: {file_path}")

        df = pd.read_csv(file_path, parse_dates=["datetime"])

        self.logger.info(
            f"{label} CSV loaded",
            extra={
                "rows": len(df),
                "columns": list(df.columns),
            }
        )

        if "datetime" not in df.columns:
            raise ValueError(f"'datetime' column missing in {file_path}")

        return df

    def _merge_data(
        self,
        load_df: pd.DataFrame,
        production_df: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge load, production and prices data on datetime.
        """
        self.logger.info("Setting datetime as index")

        load_df = load_df.set_index("datetime")
        production_df = production_df.set_index("datetime")
        prices_df = prices_df.set_index("datetime")

        self.logger.info("Performing outer joins")
        merged_df = (
            load_df
            .join(production_df, how="outer")
            .join(prices_df, how="outer")
        )

        self.logger.info(
            "Join completed",
            extra={"rows": len(merged_df)},
        )

        # Forward-fill values with lower time resolution
        if "production_ton" in merged_df.columns:
            self.logger.info("Forward-filling production_ton")
            merged_df["production_ton"] = merged_df["production_ton"].ffill()

        if "price_eur_mwh" in merged_df.columns:
            self.logger.info("Forward-filling price_eur_mwh")
            merged_df["price_eur_mwh"] = merged_df["price_eur_mwh"].ffill()

        merged_df = merged_df.reset_index()

        return merged_df
