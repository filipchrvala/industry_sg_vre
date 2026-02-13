
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

import pandas as pd
from pathlib import Path
import yaml


class SimulatePiece(BasePiece):

    def piece_function(self, input_data: InputModel) -> OutputModel:

        print("[INFO] SimulatePiece started")

        forecast_csv = Path(input_data.forecast_csv)
        solar_csv = Path(input_data.virtual_solar_csv) if input_data.virtual_solar_csv else None
        battery_csv = Path(input_data.virtual_battery_soc_csv) if input_data.virtual_battery_soc_csv else None
        scenario_yml = Path(input_data.scenario_yml)

        if not forecast_csv.exists():
            raise FileNotFoundError(f"Forecast CSV not found: {forecast_csv}")

        if not scenario_yml.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_yml}")

        print(f"[INFO] Loading forecast: {forecast_csv}")
        fc = pd.read_csv(forecast_csv, parse_dates=["datetime"])

        with open(scenario_yml) as f:
            scen = yaml.safe_load(f)

        # baseline from ML prediction
        if "prediction_load_kw" not in fc.columns:
            raise ValueError("prediction_load_kw column missing in forecast csv")

        base_load = fc["prediction_load_kw"].copy()
        simulated = base_load.copy()

        # --- SOLAR ---
        if solar_csv and solar_csv.exists():
            print("[INFO] Applying solar")
            solar_df = pd.read_csv(solar_csv, parse_dates=["datetime"])

            if "solar_kw" not in solar_df.columns:
                raise ValueError("solar_kw column missing in solar csv")

            simulated = simulated - solar_df["solar_kw"]
            simulated[simulated < 0] = 0

        # --- BATTERY (simple peak shave) ---
        if battery_csv and battery_csv.exists() and "battery" in scen:
            print("[INFO] Applying battery")
            capacity = scen["battery"].get("capacity_kwh", 0)
            max_kw = scen["battery"].get("max_kw", 0)

            soc = capacity * 0.5
            new_load = []

            for val in simulated:
                if val > max_kw and soc > 0:
                    discharge = min(val - max_kw, soc, max_kw)
                    new_load.append(val - discharge)
                    soc -= discharge
                else:
                    new_load.append(val)

            simulated = pd.Series(new_load)

        # --- PEAK SHAVING ---
        if "peak_shaving_kw" in scen:
            print("[INFO] Applying peak shaving")
            threshold = scen["peak_shaving_kw"]
            simulated[simulated > threshold] = threshold

        # --- COSTS ---
        if "price_eur_kwh" in fc.columns:
            price = fc["price_eur_kwh"].mean()
        else:
            price = 0.2

        baseline_cost = (base_load * price * 0.25).sum()
        scenario_cost = (simulated * price * 0.25).sum()
        savings = baseline_cost - scenario_cost

        # --- SAVE TIMESERIES ---
        out_df = pd.DataFrame({
            "datetime": fc["datetime"],
            "baseline_load_kw": base_load,
            "simulated_load_kw": simulated
        })

        out_path = Path(self.results_path) / "simulated_results.csv"
        out_df.to_csv(out_path, index=False)

        # --- SAVE SUMMARY ---
        summary_df = pd.DataFrame([{
            "baseline_cost_eur": baseline_cost,
            "scenario_cost_eur": scenario_cost,
            "savings_eur": savings
        }])

        summary_path = Path(self.results_path) / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("[SUCCESS] Simulation complete")

        return OutputModel(
            message="Simulation finished",
            simulated_load_csv=str(out_path),
            scenario_summary_csv=str(summary_path)
        )
