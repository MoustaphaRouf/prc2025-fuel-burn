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


def _complete(p: Path) -> bool:
    """A downloaded file is usable: zips must have their central directory, parquet files their footer."""
    if not p.exists() or p.stat().st_size == 0:
        return False
    if p.suffix == ".zip":
        return zipfile.is_zipfile(p)
    if p.suffix == ".parquet":
        with open(p, "rb") as f:
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    return True


def _fetch(src: str, p: Path, retries: int = 20, chunk: int = 1 << 20) -> None:
    """Download with resume: the large archives are often cut off mid-transfer.

    Data go to ``<name>.part``; each retry asks the server for the remaining
    bytes only (HTTP Range) and the file is renamed once its size is complete.
    """
    import time
    part = p.with_name(p.name + ".part")
    total = None
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            break
        req = urllib.request.Request(src, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 206:
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                    mode = "ab"
                else:                                   # server ignored the range: start again
                    total = int(r.headers.get("Content-Length", 0)) or None
                    mode, have = "wb", 0
                with open(part, mode) as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
            if total is None or part.stat().st_size >= total:
                break
        except Exception as e:                           # connection reset, timeout, truncated body
            print(f"{p.name}: interrupted at {part.stat().st_size if part.exists() else 0:,} bytes "
                  f"({type(e).__name__}), resuming ({attempt + 1}/{retries})")
            time.sleep(min(5 * (attempt + 1), 60))
    size = part.stat().st_size
    if total is not None and size != total:
        raise IOError(f"{p.name}: {size:,} of {total:,} bytes after {retries} attempts; run again to resume")
    part.rename(p)


def download(names, dest: str | Path = "data") -> dict[str, Path]:
    """Download the named files (keys of ``FILES``) into ``dest``.

    Files already present and complete are skipped; a truncated file left by an
    earlier interrupted run is detected and downloaded again. Interrupted
    transfers resume where they stopped.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        p = dest / FILES.get(n, n)
        if not _complete(p):
            if p.exists():
                print(f"{p.name}: incomplete file found, downloading again")
                p.unlink()
            print(f"downloading {p.name} ...")
            _fetch(url(n), p)
            if not _complete(p):
                raise IOError(f"{p.name}: downloaded file is not valid")
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
