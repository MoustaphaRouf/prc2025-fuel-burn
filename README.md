# Fuel burn per flight interval from ADS-B trajectories (PRC Data Challenge 2025)

[![tests](https://github.com/MoustaphaRouf/prc2025-fuel-burn/actions/workflows/ci.yml/badge.svg)](https://github.com/MoustaphaRouf/prc2025-fuel-burn/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MoustaphaRouf/prc2025-fuel-burn/blob/main/notebooks/reproduce_colab.ipynb)

The [EUROCONTROL PRC Data Challenge 2025](https://doi.org/10.59490/joas.2026.8750) asked participants to estimate
the fuel burnt by a flight over a given time interval [start, end], from its OpenSky ADS-B trajectory, the flight
list (aircraft type, airports, take-off and landing times) and public aircraft data. Submissions were ranked by RMSE
in kg over the ranking intervals.

This repository is the refactored pipeline of my team (*honorable-yogurt*) and an honest account of how far its
validation was from its official score.

**Best submission: 262.0 kg (v3, code not preserved). This repository reproduces v6 (263.8 kg), produced the same
day by the same pipeline.** For scale, the three winning teams scored 200.8, 212.0 and 219.9 kg.

## Method

| Step | Module | What it does |
|---|---|---|
| Trajectory cleaning | `features.clean_trajectory` | heading change wrapped to ±180° and its rate, cyclic month, forward/back fill |
| Segment features | `features.segment_features` | per interval: ground speed and vertical rate statistics, altitude, heading changes, mean longitudinal acceleration, path length, specific energy and its rate, a speed/vertical-rate stability index, cyclic time of day, fraction of flight duration, aircraft type and airports |
| Aircraft data | `aircraft.load_anp`, `aircraft.performance_table` | PRC types mapped to the closest ANP aircraft (28 types); MTOW, MLW, thrust, thrust/weight, aerodynamic coefficients averaged over phases and derived ratios |
| Physics | `aircraft.physics_features` | ISA density, dynamic pressure, bank-angle proxy and load factor, specific power |
| Encoding | `model.Encoders` | target encoding of (type, destination), origin, origin–destination pair |
| Model | `model.make_model` | XGBoost, 20,000 trees max, learning rate 0.02, depth 6, early stopping after 500 rounds |

Intervals with no trajectory point (131 of the ranking set in the original run) get no feature row. How they were
filled in the submitted file is not recorded in the notebook; `model.fallback_predictions` (interval duration ×
median fuel flow of the aircraft type) is an **addition of this repository**.

## Submission history

RMSE of each submitted file, recomputed after the challenge against the published ground truth
(`fuel_rank.parquet`, 24,289 intervals), see [`submissions/scores.csv`](submissions/scores.csv).

| Version | v1 | v2 | **v3** | v4 | v5 | **v6** | v7 | v8 | v9 | v10 | v11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RMSE (kg) | 272.1 | 274.6 | **262.0** | 264.5 | 264.5 | **263.8** | 267.3 | 265.5 | 267.4 | 299.3 | 282.0 |

## Diagnosis: 177 kg in validation, 263.8 kg in ranking

The v6 notebook reported a validation RMSE of 177.2 kg (R² 0.959). The same model scored 263.8 kg on the ranking
set, 49 % worse. Reading the code back, the validation had leaks that make it optimistic:

1. **Row-level random split.** `train_test_split` shuffled intervals, so intervals of the same flight sat on both
   sides. Within a flight, consecutive intervals share the aircraft, route, weight and weather: the model was
   largely tested on flights it had trained on. The ranking set contains only unseen flights.
2. **Early stopping on the validation set.** XGBoost stopped on the very 20 % used to report the score, so the
   number of trees was chosen on the test data.
3. **Ranking features clipped with their own quantiles.** Outlier clipping (99th percentile) was fitted separately
   on train and ranking data instead of reusing the training thresholds.
4. **Unrecorded fallback.** The model produced 23,899 predictions, fewer than the ranking intervals (24,289 in the
   ground truth); how the gap was filled in the submitted file is unknown.

Removing intervals with fuel ≥ 10,000 kg from training, on the other hand, was justified: `fuel_train` contains a
few aberrant labels (with them, the grouped validation RMSE rises to tens of thousands of kg), while the ranking
ground truth has none.

### Measured

Run in Colab on 5 November 2026, from the v6 feature files saved during the competition:

| Estimate | RMSE (kg) |
|---|---|
| Original notebook, row-level split, early stopping on validation | 177.2 |
| This repository, same procedure | 179.9 |
| **This repository, 5-fold cross-validation grouped by flight**, intervals < 10,000 kg | **230.7 ± 20.2** (folds: 226.2, 265.0, 228.5, 222.6, 211.4) |
| Official v6 submission, ranking set | 263.8 |
| This repository, v6 procedure scored on the ranking set | 265.2 |

The refactor reproduces the v6 ranking score to within 1.4 kg; the remaining gap is consistent with the different
fallback for the 131 intervals without trajectory points.

Grouping by flight closes about 60 % of the gap between validation and ranking (51 of 85 kg): the row-level split
was the main source of optimism. The remaining 35 kg are not explained by this run. Plausible causes, not tested
here: the ranking flights come from a later period than the training flights, ranking features were clipped with
their own quantiles, and the 131 fallback intervals.

Later versions (v7 to v11) did not improve the ranking score, which is what one expects when changes are judged on
a leaky validation set. `model.grouped_cv` is the estimate to use for any further change.

## Usage

```bash
git clone https://github.com/MoustaphaRouf/prc2025-fuel-burn
cd prc2025-fuel-burn
pip install -e ".[dev]"
pytest -q
```

```python
from prcfuel import aircraft, data, features, model, score

files = data.download(["fuel_train", "fuel_rank", "flightlist_train", "flightlist_rank",
                       "flights_train", "flights_rank"], "data")
fuel_train, fuel_rank = data.read_fuel(files["fuel_train"]), data.read_fuel(files["fuel_rank"])
fl_train, fl_rank = data.read_flight_list(files["flightlist_train"]), data.read_flight_list(files["flightlist_rank"])

train = features.build_dataset(data.extract(files["flights_train"], "data/train"), fuel_train, fl_train)
rank = features.build_dataset(data.extract(files["flights_rank"], "data/rank"), fuel_rank.drop(columns="fuel_kg"), fl_rank)
perfo = aircraft.performance_table(aircraft.load_anp("anp/Aircraft.csv", "anp/Aerodynamic_coefficients.csv"))

print(model.grouped_cv(train, perfo))                         # corrected validation
_, val, pred = model.reproduce_v6(train, rank, perfo)         # original procedure
sub = model.submission(fuel_rank[["idx", "flight_id", "start", "end"]], pred,
                       model.fallback_predictions(fuel_rank, fuel_train, fl_rank, fl_train))
print(val, score.score(sub, fuel_rank))
```

Command line scoring: `python -m prcfuel.score data/fuel_rank.parquet submissions/*.parquet`.

## Data and licences

- **Challenge data** (not redistributed): J. Sun, E. Spinielli, M. Strohmeier, *Aircraft Fuel Burn Estimation Data
  Challenge*, Zenodo, [doi:10.5281/zenodo.19184662](https://doi.org/10.5281/zenodo.19184662), licence
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data description: J. Sun, E. Spinielli, M. Strohmeier
  (2026), *Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge*, Journal of Open Aviation Science
  4(3), [doi:10.59490/joas.2026.8750](https://doi.org/10.59490/joas.2026.8750). `prcfuel.data.download` fetches the
  files from Zenodo.
- **ANP aircraft data** (not redistributed): download `Aircraft.csv` and `Aerodynamic_coefficients.csv` from the
  [EASA Aircraft Noise and Performance database](https://www.easa.europa.eu/en/domains/environment/policy-support-and-research/aircraft-noise-and-performance-anp-data)
  and pass their paths to `aircraft.load_anp`.
- **Code**: MIT.

## Changes from the competition notebook

- Functions renamed and grouped into modules; computations, constants and order unchanged, known quirks kept and
  marked `NOTE` (e.g. specific energy uses altitude in feet, acceleration stays in kt/s).
- A `flight_id` column is kept in the feature table for grouping; it is never a model input.
- Corrected validation (`grouped_cv`) and an explicit fallback added; the S3 upload cell and its credentials removed.
- A linear "physics" fuel estimate and CatBoost / Extra Trees / Random Forest trials from the notebook are not
  included: the submitted v6 file came from XGBoost alone.

## Résumé en français

Pipeline de l'équipe *honorable-yogurt* au PRC Data Challenge 2025 d'EUROCONTROL : estimer le carburant consommé
sur un intervalle de vol à partir de la trajectoire ADS-B. Variables de segment (cinématique, énergie spécifique,
virages, stabilité), données avion ANP, XGBoost. Meilleure soumission : 262,0 kg (v3, code non conservé). Ce dépôt
reproduit la v6 (263,8 kg), produite le même jour par le même pipeline. La validation du notebook annonçait 177 kg :
découpage aléatoire par intervalle (fuite entre intervalles d'un même vol), arrêt précoce sur l'ensemble de
validation, écrêtage ajusté sur les données de classement. Le dépôt reproduit la v6 à 1,4 kg près (265,2 kg) et
ajoute une validation croisée groupée par vol : 230,7 ± 20,2 kg, soit environ 60 % de l'écart entre validation et
classement expliqué par le découpage par intervalle. Les données (Zenodo, CC BY 4.0) et la base ANP ne sont pas redistribuées.

## License

MIT — Moustapha Amadou Roufaï
