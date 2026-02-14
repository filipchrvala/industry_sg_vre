from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

import pandas as pd
from pathlib import Path
import yaml


class SimulatePiece(BasePiece):

    def piece_function(self, input_data: InputModel) -> OutputModel:

        print("\n[INFO] ===== SIMULATE PIECE START =====")

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

        if "prediction_load_kw" not in fc.columns:
            raise ValueError("prediction_load_kw column missing in forecast csv")

        base_load = fc["prediction_load_kw"].copy()
        simulated = base_load.copy()

        print(f"[INFO] Rows in simulation: {len(fc)}")

        # ================= SOLAR =================
        if solar_csv and solar_csv.exists():
            print("[INFO] Applying solar (datetime aligned)")
            solar_df = pd.read_csv(solar_csv, parse_dates=["datetime"])

            if "solar_kw" not in solar_df.columns:
                raise ValueError("solar_kw column missing in solar csv")

            merged = pd.merge(
                fc[["datetime"]],
                solar_df[["datetime", "solar_kw"]],
                on="datetime",
                how="left"
            )

            merged["solar_kw"] = merged["solar_kw"].fillna(0)

            print(f"[DEBUG] Solar total kWh: {(merged['solar_kw'].sum()*0.25):.2f}")

            simulated = simulated - merged["solar_kw"]
            simulated[simulated < 0] = 0
        else:
            print("[INFO] No solar applied")

        # ================= BATTERY =================
        if battery_csv and battery_csv.exists() and "battery" in scen:
            print("[INFO] Applying battery")

            capacity = scen["battery"].get("capacity_kWh", 0)
            max_rate = scen["battery"].get("max_c_rate", 0) * capacity

            soc = capacity * 0.5
            new_load = []

            for val in simulated:
                if val > max_rate and soc > 0:
                    discharge = min(val - max_rate, soc, max_rate)
                    new_load.append(val - discharge)
                    soc -= discharge
                else:
                    new_load.append(val)

            simulated = pd.Series(new_load)
            print(f"[DEBUG] Battery remaining SOC kWh: {soc:.2f}")

        # ================= COST =================
        if "price_eur_kwh" not in fc.columns:
            raise ValueError("price_eur_kwh missing in forecast")

        price_series = fc["price_eur_kwh"]

        baseline_cost_series = base_load * price_series * 0.25
        scenario_cost_series = simulated * price_series * 0.25

        baseline_cost = baseline_cost_series.sum()
        scenario_cost = scenario_cost_series.sum()
        savings = baseline_cost - scenario_cost

        print("\n[DEBUG] ===== COST DEBUG =====")
        print(f"Baseline cost €: {baseline_cost:.2f}")
        print(f"Scenario cost €: {scenario_cost:.2f}")
        print(f"Savings €: {savings:.2f}")

        days = len(fc) * 15 / 60 / 24
        print(f"[DEBUG] Simulated days: {days:.1f}")

        if days < 40:
            yearly_estimate = savings * (365 / days)
            print(f"[DEBUG] Estimated yearly savings €: {yearly_estimate:.2f}")

        # ================= SAVE =================
        out_df = pd.DataFrame({
            "datetime": fc["datetime"],
            "baseline_load_kw": base_load,
            "simulated_load_kw": simulated,
            "price_eur_kwh": price_series,
            "baseline_cost_eur": baseline_cost_series,
            "scenario_cost_eur": scenario_cost_series
        })

        out_path = Path(self.results_path) / "simulated_results.csv"
        out_df.to_csv(out_path, index=False)

        summary_df = pd.DataFrame([{
            "rows": len(fc),
            "days_simulated": days,
            "baseline_cost_eur": baseline_cost,
            "scenario_cost_eur": scenario_cost,
            "savings_eur": savings,
            "estimated_yearly_savings_eur": savings * (365 / days) if days > 0 else 0
        }])

        summary_path = Path(self.results_path) / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("[SUCCESS] ===== SIMULATION COMPLETE =====")

        return OutputModel(
            message="Simulation finished (v2 realistic)",
            simulated_load_csv=str(out_path),
            scenario_summary_csv=str(summary_path)
        )
