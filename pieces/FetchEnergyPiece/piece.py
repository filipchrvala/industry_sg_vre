from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from pathlib import Path


class FetchEnergyDataPiece(BasePiece):
    """
    Domino piece responsible for:
    - loading 3 CSV files (load, production, prices)
    - merging them on datetime column
    - storing merged output as Parquet
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        """
        Main execution method called by Domino.
        """

        # Convert input paths to Path objects
        load_csv = Path(input_data.load_csv)
        production_csv = Path(input_data.production_csv)
        prices_csv = Path(input_data.prices_csv)

        # Define output path
        output_parquet = Path(self.results_path) / "merged_energy_data.parquet"

        # Validate input files
        for file_path in [load_csv, production_csv, prices_csv]:
            if not file_path.exists():
                message = f"Input file not found: {file_path}"
                self.logger.error(message)
                return OutputModel(message=message, output_path="")

        self.logger.info("Reading input CSV files")

        # Read CSV files with datetime parsing
        load_df = self._read_csv(load_csv)
        production_df = self._read_csv(production_csv)
        prices_df = self._read_csv(prices_csv)

        self.logger.info("Merging data frames")

        # Merge all data frames into one
        merged_df = self._merge_data(load_df, production_df, prices_df)

        self.logger.info("Saving merged data to Parquet")

        # Save merged data to Parquet
        merged_df.to_parquet(output_parquet, index=False)

        message = f"Data merged successfully ({len(merged_df)} rows)"

        # Set display result for Domino UI
        self.display_result = {
            "file_type": "parquet",
            "file_path": str(output_parquet)
        }

        return OutputModel(
            message=message,
            output_path=str(output_parquet)
        )

    def _read_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read CSV file and parse datetime column.
        """

        df = pd.read_csv(
            file_path,
            parse_dates=["datetime"]
        )

        return df

    def _merge_data(
        self,
        load_df: pd.DataFrame,
        production_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge load, production and prices data on datetime.
        """

        # Set datetime as index for proper time-based merge
        load_df = load_df.set_index("datetime")
        production_df = production_df.set_index("datetime")
        prices_df = prices_df.set_index("datetime")

        # Outer join keeps all timestamps
        merged_df = (
            load_df
            .join(production_df, how="outer")
            .join(prices_df, how="outer")
        )

        # Forward-fill values with lower time resolution
        if "production_ton" in merged_df.columns:
            merged_df["production_ton"] = merged_df["production_ton"].ffill()

        if "price_eur_mwh" in merged_df.columns:
            merged_df["price_eur_mwh"] = merged_df["price_eur_mwh"].ffill()

        # Reset index back to column
        merged_df = merged_df.reset_index()

        return merged_df
