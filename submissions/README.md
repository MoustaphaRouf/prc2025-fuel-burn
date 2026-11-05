# Submissions

RMSE of each submitted file, recomputed after the challenge against `fuel_rank.parquet`
(24,289 ranking intervals, Zenodo 19184662), rows matched on `idx`, with `prcfuel.score`.
Scored with `prcfuel.score` in Colab on 5 November 2026; MAE added.

The submission files themselves (`honorable-yogurt_v*.parquet`) can be added here and rescored with:

```bash
python -m prcfuel.score data/fuel_rank.parquet submissions/*.parquet
```
