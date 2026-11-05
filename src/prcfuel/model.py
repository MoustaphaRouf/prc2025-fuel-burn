"""Model, encoding and validation.

Two paths:

* :func:`reproduce_v6` repeats the competition notebook exactly: random
  row-level 80/20 split, XGBoost early-stopped on that same 20 %, test features
  clipped with their own quantiles. Its validation RMSE (177 kg in the
  notebook) is optimistic; the official score of the v6 submission was 263.8 kg.
* :func:`grouped_cv` is the corrected estimate: folds grouped by flight (no
  interval of a validation flight is seen in training), clipping thresholds and
  target encoders fitted on the training fold only, early stopping on an inner
  split of the training fold. Validation RMSE is reported on all intervals
  and on intervals below 10 000 kg: ``fuel_train`` contains a few aberrant
  labels that dominate the first figure, while the ranking ground truth has
  none, so the second one is the comparable estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aircraft import physics_features

CLIP_COLUMNS = ["Vsmean", "Vsmax", "Vsmn", "Vsstd", "Vzmax", "Vzmn", "Vzstd", "Vzmean",
                "gamma", "altmax", "altmn"]
CLIP_QUANTILE = 0.99
Y_MAX_TRAIN = 10_000
NON_FEATURES = ["y", "idx", "flight_id"]

XGB_PARAMS = dict(n_estimators=20000, learning_rate=0.02, max_depth=6, max_bin=512, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.001, reg_lambda=2.0, objective="reg:squarederror",
                  random_state=422, early_stopping_rounds=500, min_child_weight=10, gamma=0.01)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def clip_thresholds(df: pd.DataFrame, cols=CLIP_COLUMNS, q: float = CLIP_QUANTILE) -> dict:
    return {c: df[c].quantile(q) for c in cols}


def clip(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Upper-clip columns (notebook ``Corriger`` when thresholds come from ``df`` itself)."""
    df = df.copy()
    for c, hi in thresholds.items():
        df[c] = df[c].clip(None, hi)
    return df


class Encoders:
    """Target encoders of the v6 notebook: (TypeAC, Dest) jointly, Origin, OD."""

    GROUPS = (["TypeAC", "Dest"], ["Origin"], ["OD"])

    def __init__(self):
        from sklearn.preprocessing import TargetEncoder
        self.enc = [TargetEncoder() for _ in self.GROUPS]

    def fit_transform(self, x: pd.DataFrame, y) -> pd.DataFrame:
        x = x.copy()
        for e, cols in zip(self.enc, self.GROUPS):
            x[cols] = e.fit_transform(x[cols], y)   # internal cross-fitting, as in sklearn
        return _post(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for e, cols in zip(self.enc, self.GROUPS):
            x[cols] = e.transform(x[cols])
        return _post(x)


def _post(x: pd.DataFrame) -> pd.DataFrame:
    x[["altmax", "altmn"]] = x[["altmax", "altmn"]].clip(lower=0)
    cols = ["Pspec_tot", "Puissance_per_Kg", "dCAPmax", "dCAPsmax", "dCAPsmean"]
    x[cols] = x[cols].abs()
    return x


def make_model(**overrides):
    from xgboost import XGBRegressor
    return XGBRegressor(**{**XGB_PARAMS, **overrides})


def with_od(df: pd.DataFrame) -> pd.DataFrame:
    """Add the origin-destination pair if absent (the original notebook added it after loading features)."""
    if "OD" not in df.columns:
        df = df.assign(OD=df["Origin"].astype(str) + "-" + df["Dest"].astype(str))
    return df


def _xy(df: pd.DataFrame):
    return df.drop(columns=[c for c in NON_FEATURES if c in df.columns]), df["y"] if "y" in df else None


def reproduce_v6(train_feats: pd.DataFrame, test_feats: pd.DataFrame, perfo: pd.DataFrame,
                 xgb_overrides: dict | None = None):
    """Notebook procedure, unchanged. Returns (model, validation RMSE, test predictions by idx)."""
    from sklearn.model_selection import train_test_split

    train_feats, test_feats = with_od(train_feats), with_od(test_feats)
    missing = set(train_feats.columns) - set(test_feats.columns) - {"y"}
    if missing:
        raise KeyError(f"columns in training features but not in ranking features: {sorted(missing)}")
    train = clip(train_feats, clip_thresholds(train_feats))
    test = clip(test_feats, clip_thresholds(test_feats))     # NOTE: test clipped with its own quantiles
    train = train[train["y"] < Y_MAX_TRAIN]
    train, test = physics_features(train, perfo), physics_features(test, perfo)
    x, y = _xy(train)
    xtr, xva, ytr, yva = train_test_split(x, y, test_size=0.2, random_state=42)   # NOTE: row-level split
    enc = Encoders()
    xtr, xva = enc.fit_transform(xtr, ytr), enc.transform(xva)
    xt = enc.transform(_xy(test)[0][xtr.columns])
    model = make_model(**(xgb_overrides or {}))
    model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)   # NOTE: early stopping on the validation set
    val = rmse(yva, model.predict(xva))
    pred = pd.Series(model.predict(xt), index=test["idx"].values, name="fuel_kg")
    return model, val, pred


def grouped_cv(train_feats: pd.DataFrame, perfo: pd.DataFrame, n_splits: int = 5, seed: int = 0,
               xgb_overrides: dict | None = None) -> pd.DataFrame:
    """Flight-grouped cross-validation with every fitted step inside the fold."""
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    rows = []
    train_feats = with_od(train_feats)
    if train_feats["flight_id"].isna().any():
        raise ValueError("flight_id missing for some rows: grouped folds need it (see data.add_flight_id)")
    groups = train_feats["flight_id"].values
    folds = GroupKFold(n_splits=n_splits).split(train_feats, groups=groups)
    for k, (tr_i, va_i) in enumerate(folds):
        tr, va = train_feats.iloc[tr_i], train_feats.iloc[va_i]
        th = clip_thresholds(tr)
        tr, va = clip(tr, th), clip(va, th)
        tr = tr[tr["y"] < Y_MAX_TRAIN]
        tr, va = physics_features(tr, perfo), physics_features(va, perfo)
        inner_fit, inner_stop = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
                                     .split(tr, groups=tr["flight_id"]))
        fit, stop = tr.iloc[inner_fit], tr.iloc[inner_stop]
        (xf, yf), (xs, ys), (xv, yv) = _xy(fit), _xy(stop), _xy(va)
        enc = Encoders()
        xf = enc.fit_transform(xf, yf)
        xs, xv = enc.transform(xs[xf.columns]), enc.transform(xv[xf.columns])
        model = make_model(**(xgb_overrides or {}))
        model.fit(xf, yf, eval_set=[(xs, ys)], verbose=False)
        p = model.predict(xv)
        rows.append({"fold": k, "n_val": len(yv), "flights_val": va["flight_id"].nunique(),
                     "n_val_y_ge_10000": int((yv >= Y_MAX_TRAIN).sum()),
                     "rmse": rmse(yv, p), "rmse_y_lt_10000": rmse(yv[yv < Y_MAX_TRAIN], p[yv < Y_MAX_TRAIN]),
                     "best_iteration": getattr(model, "best_iteration", None)})
    return pd.DataFrame(rows)


def fallback_predictions(intervals: pd.DataFrame, fuel_train: pd.DataFrame, flight_list: pd.DataFrame,
                         flight_list_train: pd.DataFrame | None = None) -> pd.Series:
    """Predictions for intervals with no trajectory point (not in the original notebook).

    The notebook does not show how the 131+ intervals without ADS-B points were
    filled in the submitted file. This fallback is an addition of this
    repository: interval duration times the median fuel flow (kg/s) of the
    aircraft type in training, or of all training intervals if the type is unknown.
    """
    ftl = flight_list if flight_list_train is None else flight_list_train
    tr = fuel_train.merge(ftl[["flight_id", "aircraft_type"]], on="flight_id", how="left")
    dur = (pd.to_datetime(tr["end"]) - pd.to_datetime(tr["start"])).dt.total_seconds()
    tr = tr.assign(flow=tr["fuel_kg"] / dur.where(dur > 0))
    by_type = tr.groupby("aircraft_type")["flow"].median()
    overall = tr["flow"].median()
    q = intervals.merge(flight_list[["flight_id", "aircraft_type"]], on="flight_id", how="left")
    d = (pd.to_datetime(q["end"]) - pd.to_datetime(q["start"])).dt.total_seconds().clip(lower=0)
    flow = q["aircraft_type"].map(by_type).fillna(overall)
    return pd.Series((d * flow).values, index=q["idx"].values, name="fuel_kg")


def submission(intervals: pd.DataFrame, model_pred: pd.Series, fallback: pd.Series) -> pd.DataFrame:
    """Submission frame: model prediction where available, fallback elsewhere."""
    out = intervals.copy()
    out["fuel_kg"] = out["idx"].map(model_pred).astype("float64")
    missing = out["fuel_kg"].isna()
    out.loc[missing, "fuel_kg"] = out.loc[missing, "idx"].map(fallback)
    out.attrs["n_fallback"] = int(missing.sum())
    return out
