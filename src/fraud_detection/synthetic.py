"""Synthetic card transactions with a planted fraud signal.

Used by the tests and the demo. Nothing here comes from real data. The
columns follow the schema the package expects (and the original script's
``transactions_obf.csv``), and fraud probability is raised by a few
deliberately simple rules: some merchants are risky, card-not-present
entry (POS mode 81) and night-time transactions are riskier, and fraud
amounts are drawn from a heavier-tailed distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_transactions(
    n: int = 30_000,
    n_accounts: int = 800,
    n_merchants: int = 300,
    start: str = "2024-01-01",
    days: int = 240,
    base_fraud_rate: float = 0.004,
    seed: int = 0,
) -> pd.DataFrame:
    """Return ``n`` synthetic transactions sorted by time, with ``isFraud``."""
    rng = np.random.default_rng(seed)
    accounts = np.array([f"acc{i:05d}" for i in range(n_accounts)])
    merchants = np.array([f"m{i:04d}" for i in range(n_merchants)])
    merchant_mcc = rng.choice(
        ["5411", "5812", "5999", "4829", "5732", "7011"], n_merchants
    )
    merchant_country = rng.choice(
        ["826", "826", "826", "826", "442", "840"], n_merchants
    )
    risky_merchants = rng.random(n_merchants) < 0.05

    seconds = rng.uniform(0, days * 86_400, n)
    times = pd.Timestamp(start) + pd.to_timedelta(np.sort(seconds), unit="s")
    merchant_idx = rng.integers(0, n_merchants, n)
    pos_mode = rng.choice(["1", "5", "7", "81"], n, p=[0.1, 0.5, 0.2, 0.2])
    hour = times.hour.to_numpy()

    logit = (
        np.log(base_fraud_rate / (1 - base_fraud_rate))
        + 2.5 * risky_merchants[merchant_idx]
        + 1.2 * (pos_mode == "81")
        + 0.8 * ((hour >= 23) | (hour <= 5))
    )
    is_fraud = rng.random(n) < 1 / (1 + np.exp(-logit))
    amount = np.where(
        is_fraud,
        rng.lognormal(mean=4.0, sigma=1.0, size=n),
        rng.lognormal(mean=3.0, sigma=0.9, size=n),
    ).round(2)

    return pd.DataFrame(
        {
            "transactionTime": times,
            "eventId": [f"ev{i:07d}" for i in range(n)],
            "accountNumber": rng.choice(accounts, n),
            "merchantId": merchants[merchant_idx],
            "mcc": merchant_mcc[merchant_idx],
            "merchantCountry": merchant_country[merchant_idx],
            "merchantZip": rng.choice(["AB1", "CD2", "EF3", "GH4", ""], n),
            "posEntryMode": pos_mode,
            "transactionAmount": amount,
            "availableCash": rng.choice([500, 1500, 3000, 7500, 10000], n),
            "isFraud": is_fraud.astype(int),
        }
    )
