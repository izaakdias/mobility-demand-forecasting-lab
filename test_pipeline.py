from demand_forecasting.data import generate_synthetic_demand
from demand_forecasting.evaluate import TARGET, chronological_split
from demand_forecasting.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_model


def test_generator_is_deterministic_and_has_expected_contract():
    first = generate_synthetic_demand(days=4, seed=7)
    second = generate_synthetic_demand(days=4, seed=7)
    assert first.equals(second)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "timestamp"]) <= set(first.columns)
    assert first[TARGET].ge(0).all()


def test_split_is_chronological():
    train, test = chronological_split(generate_synthetic_demand(days=5))
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(test) == 5 * 24 * 4


def test_model_fits_and_predicts_without_missing_values():
    data = generate_synthetic_demand(days=8)
    train, test = chronological_split(data)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = build_model().fit(train[features], train[TARGET])
    predictions = model.predict(test[features])
    assert len(predictions) == len(test)
    assert (predictions >= 0).all()
