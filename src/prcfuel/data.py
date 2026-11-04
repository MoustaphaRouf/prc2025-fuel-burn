"""PRC Data Challenge 2025 data (Zenodo record 19184662, CC BY 4.0).

The data are not redistributed in this repository. :func:`download` fetches
the files from Zenodo; the trajectory archives are large (``flights_train.zip``
is about 3.3 GB), so fetch only what you need.

Attribution: J. Sun, E. Spinielli, M. Strohmeier, "Aircraft Fuel Burn
Estimation Data Challenge", Zenodo, https://doi.org/10.5281/zenodo.19184662
(CC BY 4.0); data description: J. Sun, E. Spinielli, M. Strohmeier (2026),
Journal of Open Aviation Science 4(3), https://doi.org/10.59490/joas.2026.8750.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZENODO_RECORD = "19184662"
FILES = {
    "fuel_train": "fuel_train.parquet",
    "fuel_rank": "fuel_rank.parquet",                    # ranking set with ground truth (published after the challenge)
    "fuel_rank_submit": "fuel_rank_submit.parquet",      # ranking intervals to predict (submission template)
    "flightlist_train": "flightlist_train.parquet",
    "flightlist_rank": "flightlist_rank.parquet",
    "flights_train": "flights_train.zip",
    "flights_rank": "flights_rank.zip",
    "readme": "readme.txt",
}


def url(name: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{FILES.get(name, name)}?download=1"


def download(names, dest: str | Path = "data") -> dict[str, Path]:
    """Download the named files (keys of ``FILES``) into ``dest``; skip those already present."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        p = dest / FILES.get(n, n)
        if not p.exists():
            urllib.request.urlretrieve(url(n), p)
        out[n] = p
    return out


def extract(zip_path: str | Path, dest: str | Path) -> list[Path]:
    """Extract a trajectory archive; return the per-flight parquet files, sorted."""
    dest = Path(dest)
    if not any(dest.rglob("*.parquet")):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    return sorted(dest.rglob("*.parquet"))


def read_fuel(path) -> pd.DataFrame:
    f = pd.read_parquet(path)
    for c in ("start", "end"):
        f[c] = pd.to_datetime(f[c])
    return f


def read_flight_list(path) -> pd.DataFrame:
    f = pd.read_parquet(path)
    for c in ("takeoff", "landed"):
        f[c] = pd.to_datetime(f[c])
    return f
