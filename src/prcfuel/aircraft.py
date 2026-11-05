"""Aircraft performance data (EASA/EUROCONTROL ANP) and physics features.

The ANP database is not redistributed here. Download ``Aircraft.csv`` and
``Aerodynamic_coefficients.csv`` from the ANP website
(https://www.easa.europa.eu/en/domains/environment/policy-support-and-research/aircraft-noise-and-performance-anp-data)
and pass their paths to :func:`load_anp`.

PRC aircraft types are mapped to the closest ANP aircraft (``ANP_MAP``),
the aerodynamic coefficients are averaged over flight phases, and a few
performance ratios are derived (notebook ``Feat_Perfo``). ``physics_features``
adds kinematic and energy quantities per segment (notebook ``Feat4_Physics``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# PRC aircraft_type -> ANP ACFT_ID (closest available model)
ANP_MAP = {
    "B789": "7878R", "A359": "A350-941", "B788": "7878R", "A332": "A330-301",
    "A21N": "A321-232", "A20N": "A320-211", "A320": "A320-211", "A333": "A330-301",
    "B739": "737800", "B738": "737800", "A321": "A321-232", "B77W": "7773ER",
    "B38M": "7378MAX", "B737": "737700", "B772": "777200", "B744": "747400",
    "B763": "767300", "A319": "A319-131", "B752": "757PW", "MD11": "MD11GE",
    "B77L": "7773ER", "A306": "A300-622R", "E75L": "EMB175", "B39M": "7378MAX",
    "A318": "A319-131", "E170": "EMB170", "A388": "A380-841", "B748": "7478",
}

ANP_COLUMNS = ["ACFT_ID", "Engine_Type", "Number Of Engines", "Weight Class",
               "Max Gross Takeoff Weight (lb)", "Max Gross Landing Weight (lb)",
               "Max Sea Level Static Thrust (lb)"]


def load_anp(aircraft_csv, aero_csv) -> pd.DataFrame:
    """One row per PRC aircraft type with ANP data and mean aerodynamic coefficients."""
    anp = pd.read_csv(aircraft_csv)[ANP_COLUMNS]
    aero = pd.read_csv(aero_csv).groupby("ACFT_ID")[["B", "C", "D", "R"]].mean().reset_index()
    aero.columns = ["ACFT_ID", "Cz_p_alpha", "Cx0", "Cxi", "Cz0"]
    m = pd.DataFrame({"aircraft_type": list(ANP_MAP), "ACFT_ID": list(ANP_MAP.values())})
    return m.merge(anp, on="ACFT_ID", how="left").merge(aero, on="ACFT_ID", how="left")


def performance_table(anp2: pd.DataFrame) -> pd.DataFrame:
    """Unit conversions and derived ratios (notebook ``Feat_Perfo``)."""
    p = anp2.copy()
    p["Weight Class"] = p["Weight Class"].map({"Large": -1, "Heavy": 1})
    p = p.rename(columns={"Max Gross Takeoff Weight (lb)": "MTOW_kg",
                          "Max Gross Landing Weight (lb)": "MLW_kg",
                          "Max Sea Level Static Thrust (lb)": "Thrust_kN"})
    p["MTOW_kg"] = p["MTOW_kg"] * 0.453592
    p["MLW_kg"] = p["MLW_kg"] * 0.453592
    p["Thrust_kN"] = p["Thrust_kN"] * 4.44822 / 1000
    p["ThrustTotal_kN"] = p["Thrust_kN"] * p["Number Of Engines"]
    g = 9.81
    p["T_W"] = p["ThrustTotal_kN"] * 1000 / (p["MTOW_kg"] * g)
    # NOTE: AR and e are proxies derived from the lift slope, not geometric values
    p["AR"] = 2 * p["Cz_p_alpha"] / (2 * np.pi - p["Cz_p_alpha"])
    p["e"] = 1.0 / (np.pi * p["Cxi"] * p["AR"])
    p["k_induced"] = 1.0 / (np.pi * p["e"] * p["AR"])
    p["Fineness_Index"] = p["Cz_p_alpha"] / (p["Cx0"] + p["Cxi"])
    p["Landing_Takeoff_Ratio"] = p["MLW_kg"] / p["MTOW_kg"]
    p = p.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return p.rename(columns={"aircraft_type": "TypeAC"}).drop(columns=["ACFT_ID", "Engine_Type"])


def physics_features(df: pd.DataFrame, perfo: pd.DataFrame) -> pd.DataFrame:
    """ISA density, dynamic pressure, load factor, specific power; merge with aircraft data."""
    df = df.copy()
    k1, k2, g = 1852 / 3600, 0.3048 / 60, 9.80665
    alt_m = df["altmn"] * 0.3048
    vs = df["Vsmean"] * k1
    vz = df["Vzmean"] * k2
    t0, lapse, r = 288.15, 0.0065, 287.05
    rho = 1.225 * (1 - lapse * alt_m / t0) ** ((g / (r * lapse)) - 1)
    qbar = 0.5 * rho * vs ** 2
    df["Pente_air"] = vz / vs
    df["VTotal"] = np.sqrt(vs ** 2 + vz ** 2)
    df["Phi"] = np.radians(df["dCAPsmean"]) * df["VTotal"] / g
    n_load = np.sqrt(1 + df["Phi"] ** 2)
    df["n_load"] = n_load
    df["Pspec_tot"] = vs * df["gamma"] + g * vz
    df["Pspec_load"] = df["Pspec_tot"] * n_load
    df["qbar"] = qbar
    df["qbar_load"] = qbar * n_load
    df["Puissance_per_Kg"] = df["VTotal"] * df["gamma"]
    df = df.merge(perfo, on="TypeAC", how="left")
    num = df.select_dtypes("number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
