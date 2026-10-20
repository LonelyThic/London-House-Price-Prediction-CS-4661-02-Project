"""Feature engineering. Every feature here only uses information from
sales that happened BEFORE the sale being predicted (no leakage)."""
import numpy as np
import pandas as pd


def month_index(df):
    """Months since Jan 1995 -> one number for 'when was it sold'."""
    return df["sale_year"] * 12 + df["sale_month"] - 1


def build_price_index(train):
    """London-wide price level per month, from training sales only.

    Uses a 12-month rolling median of log(price) to smooth out noise.
    Months after the training data (e.g. 2024) reuse the last known level.
    """
    t = month_index(train)
    monthly = np.log(train["price"]).groupby(t).median()
    all_months = pd.RangeIndex(int(t.min()), int(t.max()) + 1)
    monthly = monthly.reindex(all_months).interpolate()
    return monthly.rolling(12, min_periods=1).median()


def index_at(index, t):
    """Look up the index for months t; months past the end use the last value."""
    t = t.clip(lower=index.index.min(), upper=index.index.max())
    return t.map(index)


def add_previous_sale(train, other):
    """Add each property's most recent EARLIER sale, adjusted to today's market.

    train : sales with known prices (the only prices we're allowed to look up)
    other : the rows to add features to (validation or test set)
    Returns (train_with_features, other_with_features).
    """
    index = build_price_index(train)

    # One combined table; _row remembers the original order.
    both = pd.concat(
        [train.assign(_is_train=True),
         other.assign(_is_train=False, price=np.nan)],
        ignore_index=True,
    )
    both["_row"] = np.arange(len(both))
    both["_t"] = month_index(both)
    both["_known_price"] = both["price"].where(both["_is_train"])
    both = both.sort_values(["fullAddress", "_t"], kind="stable")

    grp = both.groupby("fullAddress", sort=False)
    both["prev_price"] = grp["_known_price"].shift()
    both["prev_t"] = grp["_t"].shift().where(both["prev_price"].notna())
    # Carry the last known sale forward past rows with no price.
    both[["prev_price", "prev_t"]] = both.groupby(
        "fullAddress", sort=False)[["prev_price", "prev_t"]].ffill()
    # A sale in the same month doesn't count as "before" this one.
    same_month = both["prev_t"] >= both["_t"]
    both.loc[same_month, ["prev_price", "prev_t"]] = np.nan

    both["months_since_prev_sale"] = both["_t"] - both["prev_t"]
    now = index_at(index, both["_t"])
    then = index_at(index, both["prev_t"].fillna(both["_t"]))
    both["prev_price_adjusted"] = both["prev_price"] * np.exp(now - then)

    cols = ["prev_price", "prev_price_adjusted", "months_since_prev_sale"]
    both = both.sort_values("_row")
    train_out, other_out = train.copy(), other.copy()
    for col in cols:
        train_out[col] = both.loc[both["_is_train"], col].to_numpy()
        other_out[col] = both.loc[~both["_is_train"], col].to_numpy()
    return train_out, other_out