"""Feature preparation and model definition."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "rainfall_mm",
    "temperature_c",
    "active_drivers",
    "avg_pickup_eta_min",
    "requests_last_hour",
    "requests_same_hour_yesterday",
]
CATEGORICAL_FEATURES = ["zone"]


def build_model(random_state: int = 42) -> Pipeline:
    """Build a preprocessing + regression pipeline with no data leakage."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    estimator = RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
