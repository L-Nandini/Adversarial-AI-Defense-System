# backend/utils.py
import os
import joblib
import pandas as pd
from typing import Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
DETECTOR_PATH = os.path.join(MODELS_DIR, "detector.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
DATA_PATH = os.path.join(DATA_DIR, "synthetic_dataset.csv")


def save_model(obj, path: str = MODEL_PATH) -> None:
    joblib.dump(obj, path)
    print(f"Saved model to {path}")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return joblib.load(path)


def save_detector(obj, path: str = DETECTOR_PATH) -> None:
    joblib.dump(obj, path)
    print(f"Saved detector to {path}")


def load_detector(path: str = DETECTOR_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return joblib.load(path)


def save_dataset(df: pd.DataFrame, path: str = DATA_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"Saved dataset to {path}")


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)
