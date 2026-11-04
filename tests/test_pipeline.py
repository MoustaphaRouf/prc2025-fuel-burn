import numpy as np
import pandas as pd
import pytest

from prcfuel import aircraft, features, model, score


def synthetic(n_flights=12, seed=0):
    """Synthetic flights: climb, cruise, descent; fuel proportional to duration and type."""
    rng = np.random.default_rng(seed)
    trajs, fuel, flist, idx = [], [], [], 0
    types = ["A320", "B789", "A21N"]
    for f in range(n_flights):
        fid = f"prc{f:04d}"
        t0 = pd.Timestamp("2025-04-12 06:00") + pd.Timedelta(hours=int(rng.integers(0, 12)))
        n = 240
        ts = t0 + pd.to_timedelta(np.arange(n) * 30, unit="s")
        alt = np.concatenate([np.linspace(0, 35000, 60), np.full(120, 35000.0), np.linspace(35000, 0, 60)])
        gs = np.concatenate([np.linspace(150, 450, 60), np.full(120, 460.0), np.linspace(450, 140, 60)])
        vr = np.gradient(alt, 0.5)  # ft/min with 30 s steps
        trk = 90 + np.cumsum(rng.normal(0, 1, n))
        trajs.append(pd.DataFrame({"timestamp": ts, "flight_id": fid, "typecode": types[f % 3],
                                   "latitude": 0.0, "longitude": 0.0, "altitude": alt, "groundspeed": gs,
                                   "track": trk, "vertical_rate": vr, "mach": np.nan, "TAS": np.nan,
                                   "CAS": np.nan, "source": "adsb"}))
        flist.append({"flight_id": fid, "aircraft_type": types[f % 3], "origin_icao": "DBBB",
                      "destination_icao": ["LFPG", "DIAP", "DNMM"][f % 3],
                      "takeoff": ts[0], "landed": ts[-1]})
        for a, b in [(0, 50), (60, 170), (180, 239)]:
            fuel.append({"idx": idx, "flight_id": fid, "start": ts[a], "end": ts[b],
                         "fuel_kg": (b - a) * 30 * (0.8 + 0.6 * (f % 3)) + rng.normal(0, 20)})
            idx += 1
        # an interval with no trajectory point (before the first point)
        fuel.append({"idx": idx, "flight_id": fid, "start": ts[0] - pd.Timedelta(minutes=20),
                     "end": ts[0] - pd.Timedelta(minutes=10), "fuel_kg": 500.0})
        idx += 1
    return trajs, pd.DataFrame(fuel), pd.DataFrame(flist)


def perfo_table():
    rows = []
    for t, acft in aircraft.ANP_MAP.items():
        heavy = t.startswith(("B7", "A3", "MD", "A388"))
        rows.append({"aircraft_type": t, "ACFT_ID": acft, "Engine_Type": "J", "Number Of Engines": 2,
                     "Weight Class": "Heavy" if heavy else "Large",
                     "Max Gross Takeoff Weight (lb)": 500000 if heavy else 170000,
                     "Max Gross Landing Weight (lb)": 380000 if heavy else 145000,
                     "Max Sea Level Static Thrust (lb)": 70000 if heavy else 27000,
                     "Cz_p_alpha": 5.5, "Cx0": 0.02, "Cxi": 0.04, "Cz0": 0.3})
    return aircraft.performance_table(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def dataset():
    trajs, fuel, flist = synthetic()
    parts = [features.segment_features(features.clean_trajectory(t), fuel, flist) for t in trajs]
    df = pd.concat(parts, ignore_index=True)
    df["OD"] = df["Origin"] + "-" + df["Dest"]
    return df, fuel, flist


def test_clean_trajectory_wraps_heading():
    t = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3, freq="10s"), "flight_id": "x",
                      "track": [350.0, 10.0, 5.0], "altitude": 0, "groundspeed": 0, "vertical_rate": 0})
    c = features.clean_trajectory(t)
    assert c["dCAP"].tolist()[1:] == [20.0, -5.0]
    assert c["dCAP_s"].iloc[1] == pytest.approx(2.0)
    assert {"SeasonSin", "SeasonCos"} <= set(c.columns) and "track" not in c.columns


def test_segment_features_shape(dataset):
    df, fuel, _ = dataset
    assert len(df) == 3 * 12            # empty intervals are skipped
    assert set(features.FEATURE_COLUMNS) <= set(df.columns)
    cruise = df.iloc[1]
    assert cruise["dZ"] == 0 and cruise["Vsmean"] == pytest.approx(460)
    assert cruise["Long_Traj"] == pytest.approx(460 * 110 * 30 / 3600)
    assert df.iloc[0]["dZ"] < 0          # climb: start minus end is negative


def test_single_point_interval_duplicated():
    trajs, fuel, flist = synthetic(1)
    t = features.clean_trajectory(trajs[0])
    one = pd.DataFrame([{"idx": 99, "flight_id": "prc0000", "start": t["timestamp"].iloc[100],
                         "end": t["timestamp"].iloc[100] + pd.Timedelta(seconds=5), "fuel_kg": 10.0}])
    out = features.segment_features(t, one, flist)
    assert len(out) == 1 and out["Vsstd"].iloc[0] == 0


def test_performance_and_physics(dataset):
    df, _, _ = dataset
    p = perfo_table()
    assert set(p["Weight Class"]) == {-1, 1}
    phys = aircraft.physics_features(df, p)
    assert len(phys) == len(df) and np.isfinite(phys.select_dtypes("number")).all().all()
    assert (phys["n_load"] >= 1).all()


def test_clip_uses_given_thresholds(dataset):
    df, _, _ = dataset
    th = model.clip_thresholds(df)
    clipped = model.clip(df, th)
    for c, hi in th.items():
        assert clipped[c].max() <= hi


def test_reproduce_and_grouped_cv(dataset):
    df, fuel, flist = dataset
    p = perfo_table()
    fast = {"n_estimators": 50, "early_stopping_rounds": 10}
    _, val, pred = model.reproduce_v6(df, df.drop(columns="y"), p, xgb_overrides=fast)
    assert val > 0 and len(pred) == len(df)
    cv = model.grouped_cv(df, p, n_splits=3, xgb_overrides=fast)
    assert len(cv) == 3 and (cv["rmse"] > 0).all()
    assert cv["flights_val"].sum() == df["flight_id"].nunique()   # each flight validated once


def test_fallback_and_submission(dataset):
    df, fuel, flist = dataset
    pred = pd.Series(df["y"].values, index=df["idx"].values)
    fb = model.fallback_predictions(fuel, fuel, flist)
    sub = model.submission(fuel[["idx", "flight_id", "start", "end"]], pred, fb)
    assert sub["fuel_kg"].notna().all() and sub.attrs["n_fallback"] == 12
    s = score.score(sub, fuel)
    assert s["matched"] == len(fuel) and s["rmse_kg"] >= 0


def test_score_matches_on_idx():
    truth = pd.DataFrame({"idx": [1, 2, 3], "fuel_kg": [100.0, 200.0, 300.0]})
    sub = pd.DataFrame({"idx": [3, 1], "fuel_kg": [310.0, 90.0]})
    s = score.score(sub, truth)
    assert s["matched"] == 2 and s["rmse_kg"] == pytest.approx(10.0)
