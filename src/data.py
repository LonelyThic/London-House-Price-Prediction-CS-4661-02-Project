from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_train():
    return pd.read_csv(DATA_DIR / "train.csv")

def load_test():
    return pd.read_csv(DATA_DIR / "test.csv")

def time_split(df, valid_from_year=2022):
    """Train on older sales, validate on recent ones (mimics predicting 2024)."""
    train = df[df["sale_year"] < valid_from_year]
    valid = df[df["sale_year"] >= valid_from_year]
    return train, valid