# Mobility Demand Forecasting Lab

An end-to-end, reproducible machine learning project for forecasting next-hour mobility demand across service zones.

This repository is a standalone portfolio project built with synthetic data. It intentionally contains no proprietary source code, customer data, or internal Leaf.app assets.

## What this demonstrates

- Time-aware feature engineering for demand forecasting
- Cyclical encoding of hour-of-day and day-of-week
- A leakage-safe chronological train/test split
- Comparison of a seasonal baseline with a scikit-learn model
- Reproducible evaluation with MAE and RMSE
- Automated tests and GitHub Actions CI

## Approach

The pipeline predicts `next_hour_requests` using operational signals that are common in mobility platforms:

- current requests and rolling request history
- active driver supply
- pickup ETA
- weather and calendar features
- service zone

The default model is a `RandomForestRegressor` wrapped in a preprocessing pipeline. The test set is always the latest portion of the generated time series, which better reflects deployment than a random split.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m demand_forecasting.train
```

Run the test suite:

```bash
pytest
```

Generate a larger experiment:

```bash
python -m demand_forecasting.train --days 180 --seed 7
```

The benchmark is synthetic and should be treated as a demonstration of engineering and modeling workflow, not as a production performance claim.

## Project structure

```text
src/demand_forecasting/
  data.py       # deterministic synthetic data generator
  features.py   # preprocessing and model pipeline
  evaluate.py   # metrics and chronological split
  train.py      # command-line experiment runner
tests/          # pipeline and data contract tests
```

## Next iterations

1. Replace synthetic data with an approved historical dataset.
2. Add walk-forward cross-validation and prediction intervals.
3. Compare tree-based models with gradient boosting and calibrated baselines.
4. Package the predictor behind a versioned inference API.
5. Monitor drift, forecast error, and supply-demand imbalance after deployment.

## License

MIT
