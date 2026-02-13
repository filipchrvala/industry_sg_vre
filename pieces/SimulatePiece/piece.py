
from domino.base_piece import BasePiece
from .models import apply_peak_shaving, simulate_solar_offset, simulate_battery_simple

import pandas as pd
from pathlib import Path
import yaml


class SimulatePiece(BasePiece):

    def piece_function(self, input_data):

        print("[INFO] SimulatePiece started")

        forecast_csv = Path(input_data["forecast_csv"])
        solar_csv = Path(input_data.get("virtual_solar_csv", ""))
        battery_csv = Path(input_data.get("virtual_battery_soc_csv", ""))
        scenario_yml = Path(input_data["scenario_yml"])

        if not forecast_csv.exists():
            raise FileNotFoundError("Forecast CSV not found")

        fc = pd.read_csv(forecast_csv, parse_dates=["datetime"])

        with open(scenario_yml) as f:
            scen = yaml.safe_load(f)

        base_load = fc["prediction_load_kw"].copy()

        simulated = base_load.copy()

        # --- solar ---
        if solar_csv.exists():
            print("[INFO] Applying solar simulation")
            solar_df = pd.read_csv(solar_csv, parse_dates=["datetime"])
            solar_series = solar_df["solar_kw"]
            simulated = simulate_solar_offset(simulated, solar_series)

        # --- battery ---
        if battery_csv.exists() and "battery" in scen:
            print("[INFO] Applying battery simulation")
            bat = simulate_battery_simple(
                simulated,
                capacity_kwh=scen["battery"]["capacity_kwh"],
                max_kw=scen["battery"]["max_kw"]
            )
            simulated = bat["grid_import_kw"]

        # --- peak shaving ---
        if "peak_shaving_kw" in scen:
            print("[INFO] Applying peak shaving")
            simulated = apply_peak_shaving(simulated, scen["peak_shaving_kw"])

        # --- costs ---
        price = fc["price_eur_kwh"].mean() if "price_eur_kwh" in fc.columns else 0.2

        baseline_cost = (base_load * price * 0.25).sum()
        scenario_cost = (simulated * price * 0.25).sum()

        savings = baseline_cost - scenario_cost

        out_df = pd.DataFrame({
            "datetime": fc["datetime"],
            "baseline_load_kw": base_load,
            "simulated_load_kw": simulated
        })

        out_path = Path(self.results_path) / "simulated_results.csv"
        out_df.to_csv(out_path, index=False)

        summary = pd.DataFrame([{
            "baseline_cost": baseline_cost,
            "scenario_cost": scenario_cost,
            "savings_eur": savings
        }])

        summary_path = Path(self.results_path) / "summary.csv"
        summary.to_csv(summary_path, index=False)

        print("[SUCCESS] Simulation complete")

        return {
            "message": "Simulation finished",
            "simulated_load_csv": str(out_path),
            "scenario_summary_csv": str(summary_path)
        }
