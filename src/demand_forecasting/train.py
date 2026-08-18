"""Run a reproducible synthetic demand forecasting experiment."""

from __future__ import annotations

import argparse

from sklearn.dummy import DummyRegressor

from .data import generate_synthetic_demand
from .evaluate import TARGET, chronological_split, regression_metrics
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_model


def run_experiment(days: int = 90, seed: int = 42) -> dict[str, dict[str, float]]:
    data = generate_synthetic_demand(days=days, seed=seed)
    train, test = chronological_split(data)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, y_train = train[features], train[TARGET]
    x_test, y_test = test[features], test[TARGET]

    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    baseline_metrics = regression_metrics(y_test, baseline.predict(x_test))

    model = build_model(random_state=seed).fit(x_train, y_train)
    model_metrics = regression_metrics(y_test, model.predict(x_test))
    return {"baseline": baseline_metrics, "random_forest": model_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = run_experiment(days=args.days, seed=args.seed)
    print(f"Synthetic chronological benchmark ({args.days} days, seed={args.seed})")
    for name, metrics in results.items():
        print(f"{name:>12} | MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f}")


if __name__ == "__main__":
    main()
