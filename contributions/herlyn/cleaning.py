"""Train-fitted cleaning; preserves raw columns, row order, IDs and prices."""
import numpy as np
import pandas as pd

NUMERIC = ['bedrooms', 'bathrooms', 'livingRooms', 'floorAreaSqM']
CATEGORICAL = ['outcode', 'tenure', 'propertyType', 'currentEnergyRating']
# Merge synonymous wording only. Keep flat subtypes and bungalow types distinct.
PROPERTY_MAP = {
    'Terraced': 'Terrace Property',
    'Mid Terrace Property': 'Mid Terrace House',
    'End Terrace Property': 'End Terrace House',
    'Semi-Detached Property': 'Semi-Detached House',
    'Detached Property': 'Detached House',
}


class PropertyCleaner:
    def __init__(self, group_rare=False, impute=False, rare_fraction=0.01):
        self.group_rare = group_rare
        self.impute = impute
        self.rare_fraction = rare_fraction

    def fit(self, train):
        tenure = train['tenure'].astype('string').str.strip()
        frequency = tenure.value_counts(normalize=True)
        self.common_tenure_ = set(frequency[frequency >= self.rare_fraction].index)
        self.medians_ = {}
        for col in NUMERIC:
            values = pd.to_numeric(train[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            values = values.mask(values <= 0)
            self.medians_[col] = float(values.median())
            if not np.isfinite(self.medians_[col]):
                raise ValueError(f'No usable training values for {col}')
        return self

    def transform(self, frame):
        if not hasattr(self, 'medians_'):
            raise RuntimeError('Fit PropertyCleaner on training data first')
        result = frame.copy()
        for col in CATEGORICAL:
            values = result[col].astype('string').str.strip()
            values = values.mask(values.str.lower().isin(['', 'none', 'nan', 'unknown', 'null']))
            if col == 'propertyType':
                values = values.replace(PROPERTY_MAP)
            if col == 'currentEnergyRating':
                values = values.str.upper()
            if col == 'tenure' and self.group_rare:
                values = values.where(values.isna() | values.isin(self.common_tenure_), 'Other')
            result[col + '_clean'] = values.fillna('Missing').astype(object)
        for col in NUMERIC:
            values = pd.to_numeric(result[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            values = values.mask(values <= 0)
            result[col + '_missing'] = values.isna().astype('int8')
            result[col + '_clean'] = values.fillna(self.medians_[col]) if self.impute else values
        result['currentEnergyRating_missing'] = (result['currentEnergyRating_clean'] == 'Missing').astype('int8')
        return result

    def fit_transform(self, train):
        return self.fit(train).transform(train)
