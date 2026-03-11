"""Streamlit dashboard app for DashboardPiece – tuned for financial / management view."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def _load_payload(path: str = "dashboard_data.json") -> dict:
    json_path = Path(path)
    if not json_path.is_file():
        # Fallback: try tests/DashboardPiece_Outputs when run from project root
        try:
            app_dir = Path(__file__).resolve().parent
            fallback = app_dir.parent.parent / "tests" / "DashboardPiece_Outputs" / "dashboard_data.json"
            if fallback.is_file():
                json_path = fallback
            else:
                return {}
        except Exception:
            return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _records_to_df(records: object) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pick_existing(columns: list[str], options: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for option in options:
        if option.lower() in lower_map:
            return lower_map[option.lower()]
    return None


def _filter_by_scenario(df: pd.DataFrame, selected_scenario: str) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["scenario", "scenario_name", "case", "variant"]:
        if col in df.columns:
            filtered = df[df[col].astype(str) == selected_scenario]
            if not filtered.empty:
                return filtered
    return df


def _first_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _kpi_value(mapping: dict, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in mapping:
            return _as_float(mapping[key], default)
    return default


def _format_eur(value: float) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:,.0f} €"


def _render_dataset_table(title: str, df: pd.DataFrame, missing_message: str) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.info(missing_message)
        return
    st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df, use_container_width=True)


def _find_soc_series(*frames: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    for frame in frames:
        if frame.empty:
            continue
        datetime_col = _pick_existing(frame.columns.tolist(), ["datetime", "timestamp", "time"])
        soc_col = _pick_existing(frame.columns.tolist(), ["soc_pct", "battery_soc", "state_of_charge", "soc"])
        if datetime_col and soc_col:
            return frame, datetime_col, soc_col
    return pd.DataFrame(), None, None


def _time_range_str(df: pd.DataFrame, dt_col: str) -> str:
    if df.empty or not dt_col or dt_col not in df.columns:
        return ""
    s = pd.to_datetime(df[dt_col], errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.min().strftime('%d.%m.%Y')} – {s.max().strftime('%d.%m.%Y')}"


def _simulation_period_days(df: pd.DataFrame, dt_col: str) -> tuple[float | None, str]:
    """Returns (period_days, range_str) for the dataframe, or (None, '')."""
    if df.empty or not dt_col or dt_col not in df.columns:
        return None, ""
    s = pd.to_datetime(df[dt_col], errors="coerce").dropna()
    if s.empty or len(s) < 2:
        return None, ""
    t_min, t_max = s.min(), s.max()
    days = (t_max - t_min).total_seconds() / 86400.0
    range_str = f"{t_min.strftime('%d.%m.%Y')} – {t_max.strftime('%d.%m.%Y')}"
    return days, range_str


# Human-readable labels for investment metrics
METRIC_LABELS = {
    "total_capex_eur": "Total investment (€)",
    "solar_capex_eur": "Solar CAPEX (€)",
    "battery_capex_eur": "Battery CAPEX (€)",
    "annual_savings_eur": "Annual savings (€)",
    "simple_payback_years": "Payback period (years)",
    "npv_eur": "Net present value (€)",
    "solar_lcoe_eur_per_mwh": "Levelized cost of energy (€/MWh)",
    "annual_co2_saved_ton": "CO₂ saved (t/year)",
    "battery_cycles_est": "Battery equivalent full cycles (over period)",
}


st.set_page_config(page_title="ISGvRE – Investment & Energy", layout="wide")
st.title("Virtual RE – Investment & Energy Dashboard")

payload = _load_payload("dashboard_data.json")
if not payload:
    st.warning("No data: dashboard_data.json was not provided or could not be parsed.")
    st.stop()

datasets = payload.get("datasets", {})
status = payload.get("inputs", {})

preprocess_df = _records_to_df(datasets.get("preprocess_predict", []))
predict_df = _records_to_df(datasets.get("predict_predictions", []))
simulate_df = _records_to_df(datasets.get("simulate_results", []))
simulate_summary_df = _records_to_df(datasets.get("simulate_summary", []))
kpi_df = _records_to_df(datasets.get("kpi_results", []))
investment_df = _records_to_df(datasets.get("investment_evaluation", []))
virtual_battery_soc_df = _records_to_df(datasets.get("virtual_battery_soc", []))

scenario_options = payload.get("scenarios") or ["Default"]
default_scenario = payload.get("default_scenario", scenario_options[0])
selected_scenario = st.selectbox(
    "Scenario",
    scenario_options,
    index=scenario_options.index(default_scenario) if default_scenario in scenario_options else 0,
)

# Solar PV & battery (from scenario YAML) – visible block
st.subheader("Solar PV & battery (scenario)")
scenario_info = payload.get("scenario_info") or {}
solar_kwp = scenario_info.get("solar_kwp")
battery_kwh = scenario_info.get("battery_kwh")
scenario_desc = (scenario_info.get("description") or "").strip()
if solar_kwp is not None or battery_kwh is not None:
    cap1, cap2 = st.columns(2)
    with cap1:
        st.metric("Solar PV capacity", f"{solar_kwp:,.0f} kWp" if solar_kwp is not None else "—")
    with cap2:
        st.metric("Battery capacity", f"{battery_kwh:,.0f} kWh" if battery_kwh is not None else "—")
    if scenario_desc:
        st.caption(scenario_desc)
else:
    st.info(
        "Solar PV and battery capacity are not in the report. "
        "Re-run the workflow so that **scenario.yml** is in DashboardPiece inputs (same file as for SimulatePiece). "
        "Then run DashboardPiece again to refresh dashboard_data.json."
    )

# Simulation period (for battery cycles and captions)
sim_period_days: float | None = None
sim_period_str: str = ""
for _df, _name in [(simulate_df, "datetime"), (virtual_battery_soc_df, "datetime"), (predict_df, "datetime")]:
    _col = _pick_existing(_df.columns.tolist(), ["datetime", "timestamp", "time"])
    if _col:
        sim_period_days, sim_period_str = _simulation_period_days(_df, _col)
        if sim_period_days is not None and sim_period_days > 0:
            break

# ----- Executive summary (for financial director) -----
st.subheader("Executive summary")
kpi_data = {}
kpi_data.update(_first_row(simulate_summary_df))
kpi_data.update(_first_row(kpi_df))
kpi_data.update(_first_row(investment_df))

total_capex = _kpi_value(kpi_data, ["total_capex_eur", "total_capex", "capex_eur"])
payback = _kpi_value(kpi_data, ["simple_payback_years", "payback_years", "payback_period", "payback"])
npv_val = _kpi_value(kpi_data, ["npv_eur", "npv", "net_present_value_eur"])
saving = _kpi_value(
    kpi_data,
    ["annual_savings_eur", "annual_savings_€", "estimated_yearly_savings_eur", "savings_eur"],
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total investment", _format_eur(total_capex))
col2.metric("Payback period", f"{payback:.1f} years" if payback and payback < 999 else "—")
col3.metric("Net present value (NPV)", _format_eur(npv_val))
col4.metric("Annual savings", _format_eur(saving))

if npv_val is not None and npv_val > 0:
    if payback and payback < 999:
        st.success(f"**Recommendation:** Positive NPV indicates the project is financially favourable. The investment is expected to pay back in about **{payback:.1f}** years.")
    else:
        st.success("**Recommendation:** Positive NPV indicates the project is financially favourable under the current assumptions.")
elif npv_val is not None and npv_val <= 0 and total_capex and total_capex > 0:
    st.info("NPV is not positive in this scenario. Consider reviewing assumptions (tariffs, CAPEX, discount rate) or timeline.")
st.divider()

# ----- Investment summary (table + short explanation) -----
st.subheader("Investment summary")
invest_display = investment_df.copy()
invest_display = _filter_by_scenario(invest_display, selected_scenario)
if not invest_display.empty:
    exclude = {"datetime", "timestamp", "date"}
    numeric_cols = [c for c in invest_display.columns if c.lower() not in exclude]
    if numeric_cols:
        row = invest_display[numeric_cols].apply(pd.to_numeric, errors="coerce").iloc[0]
        summary_data = []
        for key, val in row.dropna().items():
            label = METRIC_LABELS.get(key, key.replace("_", " ").title())
            if "eur" in key.lower() or "€" in label:
                summary_data.append({"Metric": label, "Value": _format_eur(val)})
            elif "year" in key.lower() or "payback" in key.lower():
                summary_data.append({"Metric": label, "Value": f"{val:.1f} years"})
            else:
                summary_data.append({"Metric": label, "Value": f"{val:,.2f}"})
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    # Show simulation period and battery cycles context
    if sim_period_str:
        st.caption(f"**Simulated period:** {sim_period_str}" + (f" ({sim_period_days:.0f} days)" if sim_period_days and sim_period_days > 0 else ""))
    battery_cycles_val = _kpi_value(_first_row(invest_display), ["battery_cycles_est", "cycles_equivalent"])
    if battery_cycles_val is not None and sim_period_days and sim_period_days > 0:
        cycles_per_year = battery_cycles_val * (365.0 / sim_period_days)
        st.caption(f"Battery equivalent full cycles above ({battery_cycles_val:.2f}) are for this period. **Extrapolated to one year: {cycles_per_year:.1f} equivalent full cycles/year.**")
    st.caption(
        "Positive NPV means the project is financially favourable over the analysis period. "
        "Payback is the number of years until cumulative savings cover the initial investment. "
        "Battery equivalent full cycles: total charge/discharge (SoC change) over the simulation period expressed as full 0↔100% cycles."
    )
else:
    st.info("Investment evaluation data was not provided.")
st.divider()

# ----- Consumption and cost (period + annual for financial director) -----
load_df = _filter_by_scenario(simulate_df.copy(), selected_scenario)
sim_summary_row = _first_row(_filter_by_scenario(simulate_summary_df.copy(), selected_scenario)) or _first_row(simulate_summary_df)

st.caption("**For the financial director:** Use **annual (extrapolated)** figures for planning and budgets. They are scaled from the simulated period when it is shorter than a year.")

# Period figures (consumption)
total_kwh_baseline = total_kwh_simulated = None
if not load_df.empty:
    original_col = _pick_existing(load_df.columns.tolist(), ["baseline_load_kw", "original_load_kw", "load_kw", "original_load"])
    net_col = _pick_existing(load_df.columns.tolist(), ["simulated_load_kw", "net_load_kw", "net_load", "grid_import_kw"])
    if original_col and net_col:
        total_kwh_baseline = (load_df[original_col].astype(float) * 0.25).sum()
        total_kwh_simulated = (load_df[net_col].astype(float) * 0.25).sum()

cost_baseline = _as_float(sim_summary_row.get("baseline_cost_eur")) if sim_summary_row else None
cost_scenario = _as_float(sim_summary_row.get("scenario_cost_eur")) if sim_summary_row else None
cost_savings = _as_float(sim_summary_row.get("savings_eur")) if sim_summary_row else None

# Two columns: "Over simulated period" | "Extrapolated to 1 year"
col_period, col_year = st.columns(2)
with col_period:
    st.subheader("Over simulated period")
    if sim_period_str:
        st.caption(sim_period_str + (f" ({sim_period_days:.0f} days)" if sim_period_days and sim_period_days > 0 else ""))
    if total_kwh_baseline is not None and total_kwh_simulated is not None:
        st.metric("Consumption without solar PV & battery", f"{total_kwh_baseline:,.0f} kWh")
        st.metric("Consumption with solar PV & battery", f"{total_kwh_simulated:,.0f} kWh")
    if cost_baseline is not None and cost_scenario is not None:
        st.metric("Cost without solar PV & battery", _format_eur(cost_baseline))
        st.metric("Cost with solar PV & battery", _format_eur(cost_scenario))
        st.metric("Savings", _format_eur(cost_savings))
    if total_kwh_baseline is None and cost_baseline is None:
        st.info("No consumption or cost data for this period.")

with col_year:
    st.subheader("Extrapolated to 1 year")
    st.caption("Annual equivalent (for planning). Based on simulated period.")
    if sim_period_days and sim_period_days > 0:
        factor = 365.0 / sim_period_days
        if total_kwh_baseline is not None and total_kwh_simulated is not None:
            st.metric("Consumption without solar PV & battery (per year)", f"{total_kwh_baseline * factor:,.0f} kWh/year")
            st.metric("Consumption with solar PV & battery (per year)", f"{total_kwh_simulated * factor:,.0f} kWh/year")
        if cost_baseline is not None and cost_scenario is not None and cost_savings is not None:
            st.metric("Cost without solar PV & battery (per year)", _format_eur(cost_baseline * factor))
            st.metric("Cost with solar PV & battery (per year)", _format_eur(cost_scenario * factor))
            st.metric("Savings (per year)", _format_eur(cost_savings * factor))
    else:
        st.info("Cannot extrapolate: simulated period unknown or zero.")
        if total_kwh_baseline is not None and total_kwh_simulated is not None:
            st.metric("Consumption without solar PV & battery", f"{total_kwh_baseline:,.0f} kWh")
            st.metric("Consumption with solar PV & battery", f"{total_kwh_simulated:,.0f} kWh")
        if cost_baseline is not None and cost_scenario is not None:
            st.metric("Cost without solar PV & battery", _format_eur(cost_baseline))
            st.metric("Cost with solar PV & battery", _format_eur(cost_scenario))
            st.metric("Savings", _format_eur(cost_savings))

if not sim_summary_row or "baseline_cost_eur" not in (sim_summary_row or {}):
    st.info("Cost summary (baseline / scenario) is not available – SimulatePiece output (summary.csv) required.")
st.divider()

# ----- Load curve (with time range) -----
st.subheader("Predicted consumption over time (without vs with solar PV & battery)")
if load_df.empty:
    st.info("Simulated load data was not provided.")
else:
    datetime_col = _pick_existing(load_df.columns.tolist(), ["datetime", "timestamp", "time"])
    period_str = _time_range_str(load_df, datetime_col or "")
    if period_str:
        st.caption(f"Period: {period_str}")
    original_col = _pick_existing(load_df.columns.tolist(), ["baseline_load_kw", "original_load_kw", "load_kw", "original_load"])
    net_col = _pick_existing(load_df.columns.tolist(), ["simulated_load_kw", "net_load_kw", "net_load", "grid_import_kw"])
    if datetime_col and original_col and net_col:
        chart_df = load_df[[datetime_col, original_col, net_col]].copy()
        chart_df = chart_df.sort_values(by=datetime_col)
        fig_load = px.line(
            chart_df,
            x=datetime_col,
            y=[original_col, net_col],
            title="Predicted load (kW): without vs with solar PV & battery",
        )
        fig_load.update_layout(legend_title_text="")
        st.plotly_chart(fig_load, use_container_width=True)
    else:
        st.info("Required columns for load curve are missing.")

# Cost over time (if columns available)
if not load_df.empty:
    cost_baseline_col = _pick_existing(load_df.columns.tolist(), ["baseline_cost_eur"])
    cost_scenario_col = _pick_existing(load_df.columns.tolist(), ["scenario_cost_eur"])
    dt_col_cost = _pick_existing(load_df.columns.tolist(), ["datetime", "timestamp", "time"])
    if dt_col_cost and cost_baseline_col and cost_scenario_col:
        cost_chart = load_df[[dt_col_cost, cost_baseline_col, cost_scenario_col]].copy()
        cost_chart = cost_chart.sort_values(by=dt_col_cost)
        cost_chart["Cost without solar PV & battery (€)"] = cost_chart[cost_baseline_col].astype(float)
        cost_chart["Cost with solar PV & battery (€)"] = cost_chart[cost_scenario_col].astype(float)
        fig_cost = px.line(
            cost_chart,
            x=dt_col_cost,
            y=["Cost without solar PV & battery (€)", "Cost with solar PV & battery (€)"],
            title="Predicted electricity cost over time (€ per interval)",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

# ----- Prediction (with time range) -----
st.subheader("Forecast vs actual load")
prediction_df = _filter_by_scenario(predict_df.copy(), selected_scenario)
if prediction_df.empty:
    st.info("Forecast data was not provided.")
else:
    dt_col = _pick_existing(prediction_df.columns.tolist(), ["datetime", "timestamp", "time"])
    if period_str := _time_range_str(prediction_df, dt_col or ""):
        st.caption(f"Period: {period_str}")
    actual_col = _pick_existing(prediction_df.columns.tolist(), ["load_kw", "actual_load_kw", "load"])
    pred_col = _pick_existing(prediction_df.columns.tolist(), ["prediction_load_kw", "prediction_load_mw", "predicted_load_kw"])
    if dt_col and actual_col and pred_col:
        fig_pred = px.line(
            prediction_df.sort_values(by=dt_col),
            x=dt_col,
            y=[actual_col, pred_col],
            title="Actual load vs forecast",
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Required columns for forecast chart are missing.")

# ----- Battery SoC (with time range) -----
st.subheader("Battery state of charge (SoC)")
soc_source_df, datetime_col, soc_col = _find_soc_series(
    _filter_by_scenario(virtual_battery_soc_df.copy(), selected_scenario),
    _filter_by_scenario(preprocess_df.copy(), selected_scenario),
    _filter_by_scenario(predict_df.copy(), selected_scenario),
    _filter_by_scenario(simulate_df.copy(), selected_scenario),
)
if soc_source_df.empty:
    st.info(
        "Battery SoC data was not provided. "
        "Ensure **virtual_battery_soc.csv** (BatterySimPiece output) is passed to DashboardPiece and the workflow has been run so that dashboard_data.json contains the battery SoC series."
    )
else:
    if datetime_col:
        st.caption(f"Period: {_time_range_str(soc_source_df, datetime_col)}")
    soc_df = soc_source_df[[datetime_col, soc_col]].copy().sort_values(by=datetime_col)
    fig_soc = px.line(soc_df, x=datetime_col, y=soc_col, title="Battery SoC (%)")
    st.plotly_chart(fig_soc, use_container_width=True)

# ----- Investment metrics (bar chart, human-readable labels) -----
st.subheader("Investment metrics (chart)")
invest_df = _filter_by_scenario(investment_df.copy(), selected_scenario)
if invest_df.empty:
    st.info(
        "Investment evaluation data was not provided. "
        "Ensure **investment_evaluation.csv** (InvestmentEvalPiece output) is passed to DashboardPiece and the workflow has been run."
    )
else:
    exclude = {"datetime", "timestamp", "date"}
    numeric_cols = [c for c in invest_df.columns if c.lower() not in exclude]
    if not numeric_cols:
        st.info("No numeric columns in investment_evaluation data – cannot draw chart.")
    else:
        row = invest_df[numeric_cols].apply(pd.to_numeric, errors="coerce").iloc[0]
        metrics_df = row.dropna().reset_index()
        metrics_df.columns = ["metric", "value"]
        metrics_df["label"] = metrics_df["metric"].map(lambda x: METRIC_LABELS.get(x, x.replace("_", " ").title()))
        if metrics_df.empty:
            st.info("No numeric values in investment_evaluation data – cannot draw chart.")
        else:
            fig_inv = px.bar(metrics_df, x="label", y="value", title="Investment evaluation")
            st.plotly_chart(fig_inv, use_container_width=True)

# ----- Technical data (expander) -----
with st.expander("Technical data – source files and status"):
    st.subheader("Source file data")
    _render_dataset_table(
        "PreprocessEnergyDataPiece: predict_dataset_15min.parquet",
        preprocess_df,
        "File not provided or empty.",
    )
    _render_dataset_table("PredictPiece: predictions_15min.csv", predict_df, "File not provided or empty.")
    _render_dataset_table("SimulatePiece: simulated_results.csv", simulate_df, "File not provided or empty.")
    _render_dataset_table("SimulatePiece: summary.csv", simulate_summary_df, "File not provided or empty.")
    _render_dataset_table("BatterySimPiece: virtual_battery_soc.csv", virtual_battery_soc_df, "File not provided or empty.")
    _render_dataset_table("KPIPiece: kpi_results.csv", kpi_df, "File not provided or empty.")
    _render_dataset_table("InvestmentEvalPiece: investment_evaluation.csv", investment_df, "File not provided or empty.")
    st.subheader("Input files status")
    status_rows = []
    for input_name, details in status.items():
        status_rows.append({
            "input": input_name,
            "provided": details.get("provided", False),
            "rows": details.get("rows", 0),
            "error": details.get("error"),
        })
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No input status metadata available.")
