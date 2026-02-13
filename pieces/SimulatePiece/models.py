
import pandas as pd
import numpy as np


def apply_peak_shaving(load: pd.Series, threshold_kw: float):
    """Reduce peaks above threshold (simple clipping)."""
    shaved = load.copy()
    shaved[shaved > threshold_kw] = threshold_kw
    return shaved


def simulate_solar_offset(load: pd.Series, solar: pd.Series):
    """Subtract solar from load, cannot go below zero."""
    net = load - solar
    net[net < 0] = 0
    return net


def simulate_battery_simple(net_load: pd.Series, capacity_kwh: float,
                            max_kw: float, eff: float = 0.95):
    """Very simple battery peak shaving."""
    soc = capacity_kwh * 0.5  # start 50%
    soc_list = []
    grid = []

    for val in net_load:
        # discharge if high load
        if val > max_kw and soc > 0:
            discharge = min(val - max_kw, soc, max_kw)
            discharge *= eff
            grid.append(val - discharge)
            soc -= discharge
        else:
            grid.append(val)

        soc_list.append(soc)

    return pd.DataFrame({
        "grid_import_kw": grid,
        "battery_soc_kwh": soc_list
    })
