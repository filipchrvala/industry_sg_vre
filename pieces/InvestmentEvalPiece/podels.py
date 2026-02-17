import numpy as np


def simple_payback(capex: float, annual_savings: float) -> float:
    if annual_savings <= 0:
        return np.inf
    return capex / annual_savings


def npv(capex: float, annual_cash: float, years: int, dr: float) -> float:
    """Net present value"""
    if dr == 0:
        return -capex + annual_cash * years

    factor = (1 - (1 + dr) ** -years) / dr
    return -capex + annual_cash * factor


def co2_saved(annual_pv_mwh: float, grid_factor: float = 0.35) -> float:
    """EU realistic grid factor"""
    return annual_pv_mwh * grid_factor


def lcoe(capex: float, annual_mwh: float, degradation: float, years: int) -> float:
    """Levelised cost of energy"""
    if annual_mwh <= 0:
        return np.inf

    total_mwh = sum(annual_mwh * (1 - degradation) ** y for y in range(years))
    return capex / total_mwh if total_mwh else np.inf
