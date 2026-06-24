#!/usr/bin/env python3
"""
Trains a RandomForest classifier on the synthetic dataset and saves the model
and anomaly detector.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_FILE = "backend/data/synthetic_dataset.csv"
MODEL_DIR = "backend/models"
MODEL_FILE = "model.pkl"
DETECTOR_FILE = "detector.pkl"  # optional detector placeholder

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(csv_path=DATA_FILE):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df[[c for c in df.columns if c.startswith("f")]].values
    y = df["label"].values
    return X, y


def train_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Dummy detector placeholder
    detector_path = os.path.join(MODEL_DIR, DETECTOR_FILE)
    with open(detector_path, "wb") as f:
        pickle.dump({"detector": "dummy"}, f)

    acc = clf.score(X_test, y_test)
    print(f"Model trained and saved at {model_path}")
    print(f"Accuracy on test set: {acc:.4f}")


if __name__ == "__main__":
    train_model()
