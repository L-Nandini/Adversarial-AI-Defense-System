# backend/generate_data.py
#!/usr/bin/env python3
"""
Generates a richer synthetic dataset with multiple adversarial/noise styles.
Saves to backend/data/synthetic_dataset.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

DATA_DIR = "backend/data"
CSV_FILE = "synthetic_dataset.csv"

def generate_synthetic_data(
    n_samples: int = 6000,
    n_features: int = 16,
    n_informative: int = 10,
    random_state: int = 42
):
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, CSV_FILE)

    rng = np.random.default_rng(random_state)

    # Base dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=1.2,
        flip_y=0.01,
        random_state=random_state
    )

    base = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    base["label"] = y
    base["source"] = "normal"

    # Noisy samples (Gaussian)
    n_noisy = n_samples // 6
    idx = rng.choice(n_samples, size=n_noisy, replace=False)
    noisy_X = X[idx] + rng.normal(0, 0.6, size=(n_noisy, n_features))
    noisy = pd.DataFrame(noisy_X, columns=[f"f{i}" for i in range(n_features)])
    noisy["label"] = y[idx]
    noisy["source"] = "noisy_gaussian"

    # Adversarial style 1: shifted perturbation
    adv1_X = X[idx] + rng.normal(2.5, 1.0, size=(n_noisy, n_features))
    adv1 = pd.DataFrame(adv1_X, columns=[f"f{i}" for i in range(n_features)])
    adv1["label"] = y[idx]
    adv1["source"] = "adversarial_shift"

    # Adversarial style 2: sparse spikes (few features get huge changes)
    adv2_X = X[idx].copy()
    spikes = rng.integers(1, max(2, n_features // 4), size=n_noisy)  # how many features spiked per row
    for i in range(n_noisy):
        cols = rng.choice(n_features, size=spikes[i], replace=False)
        adv2_X[i, cols] += rng.normal(0, 8.0, size=len(cols))
    adv2 = pd.DataFrame(adv2_X, columns=[f"f{i}" for i in range(n_features)])
    adv2["label"] = y[idx]
    adv2["source"] = "adversarial_sparse_spike"

    # Adversarial style 3: sign-flip + magnitude scaling
    adv3_X = X[idx] * rng.uniform(1.8, 3.0, size=(n_noisy, 1))
    adv3_X[:, : n_features // 3] *= -1
    adv3 = pd.DataFrame(adv3_X, columns=[f"f{i}" for i in range(n_features)])
    adv3["label"] = y[idx]
    adv3["source"] = "adversarial_flip_scale"

    # Outliers (uniform)
    n_out = n_samples // 6
    out_X = rng.uniform(-40, 40, size=(n_out, n_features))
    out = pd.DataFrame(out_X, columns=[f"f{i}" for i in range(n_features)])
    out["label"] = rng.integers(0, 2, size=n_out)
    out["source"] = "outlier_uniform"

    # Combine
    final_df = pd.concat([base, noisy, adv1, adv2, adv3, out], ignore_index=True)
    final_df.to_csv(csv_path, index=False)
    print(f"Dataset created at {csv_path} with {len(final_df)} rows")

if __name__ == "__main__":
    generate_synthetic_data()
