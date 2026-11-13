"""Tiny synthetic BankSim-shaped data for tests (no Kaggle download needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BANKSIM_CSV_NAME = "bs140513_032310.csv"
CATEGORIES = (
    "es_barsandrestaurants",
    "es_contents",
    "es_fashion",
    "es_food",
    "es_health",
    "es_home",
    "es_hotelservices",
    "es_hyper",
    "es_leisure",
    "es_otherservices",
    "es_sportsandtoys",
    "es_tech",
    "es_transportation",
    "es_travel",
    "es_wellnessandbeauty",
)
# Fraud concentrates in a few categories with inflated amounts, as in the real simulator.
RISKY_CATEGORIES = ("es_leisure", "es_travel", "es_sportsandtoys", "es_hotelservices")


def synthetic_banksim(n_customers: int = 120, seed: int = 0, days: int = 180) -> pd.DataFrame:
    """BankSim-shaped transactions: quoted ids like the real CSV, ~1% fraud, several rows per day."""
    rng = np.random.default_rng(seed)
    merchants = [f"'M{1000 + i}'" for i in range(30)]
    merchant_category = {m: CATEGORIES[i % len(CATEGORIES)] for i, m in enumerate(merchants)}
    base_amount = {c: float(rng.uniform(15, 60)) for c in CATEGORIES}

    frames = []
    for customer in range(n_customers):
        n = int(max(5, rng.poisson(60)))
        step = np.sort(rng.integers(0, days, size=n))
        favourites = rng.choice(len(merchants), size=6, replace=False)
        merchant = np.array(merchants)[rng.choice(favourites, size=n)]
        category = np.array([merchant_category[m] for m in merchant])
        amount = np.array([rng.lognormal(np.log(base_amount[c]), 0.5) for c in category])
        fraud = np.zeros(n, dtype=int)
        if customer % 4 == 0:  # a quarter of the customers are targeted
            risky = np.flatnonzero(np.isin(category, RISKY_CATEGORIES))
            hit = risky[rng.random(len(risky)) < 0.35]
            fraud[hit] = 1
            amount[hit] *= rng.uniform(6, 12, size=len(hit))
        frames.append(
            pd.DataFrame(
                {
                    "step": step,
                    "customer": f"'C{customer}'",
                    "age": f"'{customer % 7}'",
                    "gender": "'F'" if customer % 2 else "'M'",
                    "zipcodeOri": "'28007'",
                    "merchant": merchant,
                    "zipMerchant": "'28007'",
                    "category": [f"'{c}'" for c in category],
                    "amount": np.round(amount, 2),
                    "fraud": fraud,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    # The real CSV is not time-ordered; shuffle rows so nothing relies on file order.
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_synthetic_csv(project_root: Path, **kwargs: object) -> pd.DataFrame:
    """Write the synthetic frame where `DatasetLoader` expects the BankSim CSV; return it."""
    frame = synthetic_banksim(**kwargs)
    target = project_root / "data" / "raw" / "banksim"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / BANKSIM_CSV_NAME, index=False)
    return frame
