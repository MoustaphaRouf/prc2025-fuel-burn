"""Segment features from ADS-B trajectories (submission v6 pipeline).

For every fuel interval [start, end] of a flight, the trajectory points that
fall inside the interval are summarised into one row: ground speed, altitude,
heading change, vertical rate, kinematics, specific energy, flight stability,
time of day and season, plus the flight's aircraft type and airports.

This is a faithful refactor of the competition notebook (functions ``Feat1``
and ``Feat_Final``): the computations and their order are unchanged, so the
features match the ones the v6 model was trained on. Known quirks are kept on
purpose and flagged with ``NOTE``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.80665          # m/s²
KT_TO_MS = 1852 / 3600
FTMIN_TO_MS = 0.3048 / 60

FEATURE_COLUMNS = [
    "Long_Traj", "dZ", "Vsmean", "Vsmax", "Vsmn", "Vsstd", "gamma", "altmax", "altmn",
    "dCAPmax", "dCAPstd", "dCAPtotal", "dCAPsmax", "dCAPsmean", "dCAPsstd",
    "Vzmax", "Vzmn", "Vzstd", "Vzmean", "SeasonSinmean", "SeasonCosmean",
    "TypeAC", "Dest", "Origin", "Fract_dt", "TSin", "TCos", "Turn_eff", "Pente",
    "Energie", "dE", "dE_std", "Stability", "Stability_std",
]


def clean_trajectory(flight: pd.DataFrame) -> pd.DataFrame:
    """First pass on one flight's trajectory (notebook ``Feat1``).

    Drops near-empty columns, derives heading change (wrapped to ±180°) and its
    rate, encodes the month cyclically, and forward/back-fills gaps.
    """
    t = flight.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t = t.drop(columns=[c for c in ("mach", "TAS", "CAS", "source") if c in t.columns])
    t["dCAP"] = (t["track"].diff() + 180) % 360 - 180
    t = t.drop(columns=["track"])
    t["dt"] = t["timestamp"].diff().dt.total_seconds()
    t["dCAP_s"] = t["dCAP"] / t["dt"]
    month = t["timestamp"].dt.month
    t["SeasonSin"] = np.sin(2 * np.pi / 12 * month)
    t["SeasonCos"] = np.cos(2 * np.pi / 12 * month)
    return t.ffill().bfill()


def _div(a, b) -> float:
    """Float division with numpy semantics (inf or NaN on zero), as in the notebook."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _time_of_day(takeoff: pd.Timestamp, landed: pd.Timestamp) -> float:
    """Mid-flight hour of day, in [0, 24)."""
    t1 = takeoff.hour + takeoff.minute / 60
    t2 = landed.hour + landed.minute / 60
    if t2 < t1:
        t2 += 24
    tm = (t1 + t2) / 2
    return tm - 24 if tm > 24 else tm


def segment_features(traj: pd.DataFrame, fuel: pd.DataFrame, flight_list: pd.DataFrame) -> pd.DataFrame:
    """One row per fuel interval of this flight (notebook ``Feat_Final``).

    ``traj`` is the output of :func:`clean_trajectory` for a single flight;
    ``fuel`` has ``idx, flight_id, start, end`` (and ``fuel_kg`` for training);
    ``flight_list`` has ``flight_id, aircraft_type, origin_icao,
    destination_icao, takeoff, landed``.

    Intervals with no trajectory point are skipped (they need a fallback, see
    :func:`prcfuel.model.fallback_predictions`). An interval with a single point
    is duplicated so that spreads are defined (NOTE: their std is then 0).
    """
    flight_id = traj["flight_id"].iloc[0]
    segs = fuel[fuel["flight_id"] == flight_id]
    meta = flight_list[flight_list["flight_id"] == flight_id].iloc[0]
    total_h = (meta["landed"] - meta["takeoff"]).total_seconds() / 3600
    tm = _time_of_day(meta["takeoff"], meta["landed"])
    rows = []
    for seg in segs.itertuples(index=False):
        start, end = seg.start, seg.end
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        s = traj[(traj["timestamp"] >= start) & (traj["timestamp"] <= end)]
        if s.empty:
            continue
        if len(s) == 1:
            s = pd.concat([s, s], ignore_index=True)
        vs = s["groundspeed"]
        hours = (s["timestamp"].iloc[-1] - s["timestamp"].iloc[0]).total_seconds() / 3600
        length = vs.mean() * hours                      # NM (ground-speed integral approximation)
        dz = s["altitude"].iloc[0] - s["altitude"].iloc[-1]   # NOTE: start minus end (positive = descent)
        dcap_total = s["dCAP"].abs().sum()
        vs_ms = vs * KT_TO_MS
        e_spec = G * s["altitude"].astype(float) + 0.5 * vs_ms ** 2   # NOTE: altitude in ft, as in the notebook
        de = e_spec.diff().ffill().bfill() / s["dt"].astype(float).replace(0, np.nan)
        vz_ms = s["vertical_rate"].astype(float) * FTMIN_TO_MS
        stability = vs_ms.rolling(5, min_periods=1).std() + vz_ms.rolling(5, min_periods=1).std()
        row = {
            "idx": seg.idx, "flight_id": flight_id,
            "Long_Traj": length, "dZ": dz,
            "Vsmean": vs.mean(), "Vsmax": vs.max(), "Vsmn": vs.min(), "Vsstd": vs.std(),
            "gamma": (vs.diff() / s["dt"]).mean(),     # NOTE: kt/s, not converted
            "altmax": s["altitude"].max(), "altmn": s["altitude"].min(),
            "dCAPmax": s["dCAP"].max(), "dCAPstd": s["dCAP"].std(), "dCAPtotal": dcap_total,
            "dCAPsmax": s["dCAP_s"].max(), "dCAPsmean": s["dCAP_s"].mean(), "dCAPsstd": s["dCAP_s"].std(),
            "Vzmax": s["vertical_rate"].max(), "Vzmn": s["vertical_rate"].min(),
            "Vzstd": s["vertical_rate"].std(), "Vzmean": s["vertical_rate"].mean(),
            "SeasonSinmean": s["SeasonSin"].mean(), "SeasonCosmean": s["SeasonCos"].mean(),
            "TypeAC": meta["aircraft_type"], "Dest": meta["destination_icao"], "Origin": meta["origin_icao"],
            "Fract_dt": _div(hours, total_h),
            "TSin": np.sin(2 * np.pi / 24 * tm), "TCos": np.cos(2 * np.pi / 24 * tm),
            "Turn_eff": _div(dcap_total, length),   # inf/NaN when length is 0, set to 0 downstream
            "Pente": _div(dz, length),
            "Energie": e_spec.mean(), "dE": de.mean(), "dE_std": de.std(),
            "Stability": stability.mean(), "Stability_std": stability.std(),
        }
        if "fuel_kg" in segs.columns:
            row["y"] = seg.fuel_kg
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(trajectory_files, fuel: pd.DataFrame, flight_list: pd.DataFrame, progress: bool = True) -> pd.DataFrame:
    """Segment features for every trajectory file (one parquet per flight)."""
    it = trajectory_files
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(trajectory_files)
        except ImportError:
            pass
    parts = []
    for path in it:
        traj = clean_trajectory(pd.read_parquet(path))
        f = segment_features(traj, fuel, flight_list)
        if not f.empty:
            parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    out["OD"] = out["Origin"] + "-" + out["Dest"]
    return out
