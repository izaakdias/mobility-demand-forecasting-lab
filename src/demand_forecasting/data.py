"""Synthetic mobility demand data for reproducible portfolio experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_demand(days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Create deterministic, hourly zone-level demand observations.

    The generator models common mobility patterns without using any real
    customer, driver, location, or company data.
    """
    if days < 3:
        raise ValueError("days must be at least 3")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    zones = ("central", "north", "south", "west")
    rows: list[dict[str, object]] = []

    zone_effect = {"central": 13.0, "north": 7.0, "south": 5.0, "west": 9.0}

    for zone in zones:
        for timestamp in timestamps:
            hour = timestamp.hour
            weekday = timestamp.dayofweek
            rush_hour = float(1.0 <= hour <= 3 or 7 <= hour <= 9 or 17 <= hour <= 19)
            evening_peak = float(17 <= hour <= 20)
            weekend = float(weekday >= 5)
            rain_mm = max(0.0, rng.normal(1.4 if weekend else 0.8, 1.8))
            temperature_c = 24.0 + 4.0 * np.sin((hour - 7) * np.pi / 12) + rng.normal(0, 1.2)
            current_requests = max(
                0.0,
                zone_effect[zone]
                + 10.0 * rush_hour
                + 5.0 * evening_peak
                + 3.0 * weekend
                + 1.6 * rain_mm
                + rng.normal(0, 2.0),
            )
            active_drivers = max(1.0, 25.0 + 8.0 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 2.5))
            avg_pickup_eta = max(2.0, 4.0 + current_requests / max(active_drivers, 1.0) * 3.0 + rng.normal(0, 0.5))
            historical_requests_1h = max(0.0, current_requests + rng.normal(0, 2.0))
            historical_requests_24h = max(0.0, current_requests + 2.0 * rush_hour + rng.normal(0, 3.0))
            next_hour_rush = float(1 <= (hour + 1) % 24 <= 3 or 7 <= (hour + 1) % 24 <= 9 or 17 <= (hour + 1) % 24 <= 19)
            next_hour_requests = max(
                0.0,
                zone_effect[zone]
                + 10.0 * next_hour_rush
                + 5.0 * float(17 <= (hour + 1) % 24 <= 20)
                + 3.0 * weekend
                + 1.6 * rain_mm
                + 0.25 * historical_requests_1h
                + rng.normal(0, 2.0),
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "zone": zone,
                    "hour": hour,
                    "day_of_week": weekday,
                    "is_weekend": int(weekend),
                    "rainfall_mm": round(rain_mm, 3),
                    "temperature_c": round(temperature_c, 3),
                    "active_drivers": round(active_drivers, 3),
                    "avg_pickup_eta_min": round(avg_pickup_eta, 3),
                    "requests_last_hour": round(historical_requests_1h, 3),
                    "requests_same_hour_yesterday": round(historical_requests_24h, 3),
                    "next_hour_requests": round(next_hour_requests, 3),
                }
            )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
