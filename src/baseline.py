"""Baseline models. Run from the repo root:  python -m src.baseline"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.data import load_train, time_split

CAT_COLS = ["outcode", "tenure", "propertyType", "currentEnergyRating"]
NUM_COLS = ["latitude", "longitude", "bathrooms", "bedrooms",
            "floorAreaSqM", "livingRooms", "sale_month", "sale_year"]


def encode(train, other):
    """Convert text categories to integer codes, using only categories seen in training."""
    X_train = train[CAT_COLS + NUM_COLS].copy()
    X_other = other[CAT_COLS + NUM_COLS].copy()
    for col in CAT_COLS:
        cats = pd.Categorical(train[col]).categories
        for X, src in ((X_train, train), (X_other, other)):
            codes = pd.Categorical(src[col], categories=cats).codes.astype(float)
            codes[codes < 0] = np.nan          # missing / unseen -> NaN
            X[col] = codes
    return X_train, X_other


def median_baseline(train, valid):
    """Predict the recent median price for the same outcode and property type."""
    recent = train[train["sale_year"] >= train["sale_year"].max() - 2]
    medians = recent.groupby(["outcode", "propertyType"])["price"].median()
    keys = pd.MultiIndex.from_frame(valid[["outcode", "propertyType"]])
    preds = pd.Series(medians.reindex(keys).values, index=valid.index)
    return preds.fillna(recent["price"].median())


def gradient_boosting(train, valid):
    """Gradient boosting trained on log(price), converted back to pounds."""
    X_train, X_valid = encode(train, valid)
    model = HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.08, max_leaf_nodes=127,
        categorical_features=list(range(len(CAT_COLS))), random_state=42,
    )
    model.fit(X_train, np.log(train["price"]))
    return np.exp(model.predict(X_valid))


if __name__ == "__main__":
    train, valid = time_split(load_train(), valid_from_year=2023)
    print(f"Train: {len(train):,} rows (before 2023) | Valid: {len(valid):,} rows (2023)")

    for name, fn in [("Median by outcode + type", median_baseline),
                     ("Gradient boosting", gradient_boosting)]:
        mae = mean_absolute_error(valid["price"], fn(train, valid))
        print(f"{name:<28} MAE: £{mae:,.0f}")