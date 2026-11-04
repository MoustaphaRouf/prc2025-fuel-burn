"""Official-style scoring of a submission against the ranking ground truth.

The challenge ranked submissions by RMSE (kg) on the ranking intervals. Rows
are matched on ``idx`` only: ``start``/``end`` dtypes differ between files.

    python -m prcfuel.score data/fuel_rank.parquet submissions/*.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def score(submission: pd.DataFrame, truth: pd.DataFrame) -> dict:
    m = submission[["idx", "fuel_kg"]].merge(truth[["idx", "fuel_kg"]], on="idx", how="right",
                                             suffixes=("_pred", "_true"))
    ok = m["fuel_kg_pred"].notna()
    err = m.loc[ok, "fuel_kg_pred"] - m.loc[ok, "fuel_kg_true"]
    return {"rmse_kg": float(np.sqrt(np.mean(err ** 2))), "matched": int(ok.sum()), "truth_rows": len(truth),
            "mae_kg": float(np.mean(np.abs(err)))}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    truth = pd.read_parquet(argv[0])
    rows = [{"file": Path(p).name, **score(pd.read_parquet(p), truth)} for p in argv[1:]]
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
