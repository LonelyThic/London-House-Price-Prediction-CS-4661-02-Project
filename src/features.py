"""Feature engineering. Every feature here only uses information from
sales that happened BEFORE the sale being predicted (no leakage)."""
import numpy as np
import pandas as pd


def month_index(df):
    """Months since Jan 1995 -> one number for 'when was it sold'."""
    return df["sale_year"] * 12 + df["sale_month"] - 1


def build_price_index(train):
    """London-wide price level per month, from training sales only.

    Each month's value uses only EARLIER months (12-month rolling median of
    log(price), shifted by one month), so a sale never sees its own month.
    Months after the training data (e.g. 2024) reuse the last known level.
    """
    t = month_index(train)
    monthly = np.log(train["price"]).groupby(t).median()
    all_months = pd.RangeIndex(int(t.min()), int(t.max()) + 2)
    monthly = monthly.reindex(all_months).interpolate(limit_area="inside")
    smooth = monthly.rolling(12, min_periods=1).median().shift(1)
    return smooth.bfill()


def index_at(index, t):
    """Look up the index for months t; months past the end use the last value."""
    t = t.clip(lower=index.index.min(), upper=index.index.max())
    return t.map(index)


def add_previous_sale(train, other):
    """Add each property's most recent sale from an EARLIER month.

    train : sales with known prices (the only prices we're allowed to look up)
    other : the rows to add features to (validation or test set)
    Returns (train_with_features, other_with_features).
    """
    index = build_price_index(train)

    both = pd.concat(
        [train.assign(_is_train=True),
         other.assign(_is_train=False, price=np.nan)],
        ignore_index=True,
    )
    both["_t"] = month_index(both)
    both["_known_price"] = both["price"].where(both["_is_train"])

    # One row per (address, month): the last known price that month.
    # Shifting at the month level means same-month sales never count,
    # and a property falls back to its sale from an earlier month.
    monthly = (both.groupby(["fullAddress", "_t"])["_known_price"]
                   .last().reset_index())
    monthly["_sale_t"] = monthly["_t"].where(monthly["_known_price"].notna())
    grp = monthly.groupby("fullAddress", sort=False)
    monthly["prev_price"] = grp["_known_price"].shift()
    monthly["prev_t"] = grp["_sale_t"].shift()
    monthly[["prev_price", "prev_t"]] = monthly.groupby(
        "fullAddress", sort=False)[["prev_price", "prev_t"]].ffill()

    both = both.merge(
        monthly[["fullAddress", "_t", "prev_price", "prev_t"]],
        on=["fullAddress", "_t"], how="left", validate="many_to_one")

    both["months_since_prev_sale"] = both["_t"] - both["prev_t"]
    now = index_at(index, both["_t"])
    then = index_at(index, both["prev_t"].fillna(both["_t"]))
    both["prev_price_adjusted"] = both["prev_price"] * np.exp(now - then)

    cols = ["prev_price", "prev_price_adjusted", "months_since_prev_sale"]
    train_out, other_out = train.copy(), other.copy()
    for col in cols:
        train_out[col] = both.loc[both["_is_train"], col].to_numpy()
        other_out[col] = both.loc[~both["_is_train"], col].to_numpy()
    return train_out, other_out