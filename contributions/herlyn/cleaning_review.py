"""Run from repository root: python -m contributions.herlyn.cleaning_review.

Uses the original model's hyperparameters, pre-2023 training and 2023 validation.
Does not change Steven's previous-sale implementation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from src.data import load_train, time_split
from src.baseline import gradient_boosting, median_baseline, NUM_COLS
from contributions.herlyn.cleaning import PropertyCleaner, NUMERIC, CATEGORICAL

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'reports'


def predict_clean(train, valid, group_rare=False, impute=False):
    cleaner = PropertyCleaner(group_rare=group_rare, impute=impute).fit(train)
    a, b = cleaner.transform(train), cleaner.transform(valid)
    nums = [c + '_clean' if c in NUMERIC else c for c in NUM_COLS]
    nums += [c + '_missing' for c in NUMERIC] + ['currentEnergyRating_missing']
    cats = [c + '_clean' for c in CATEGORICAL]
    xa, xb = a[cats + nums].copy(), b[cats + nums].copy()
    for c in cats:
        levels = pd.Categorical(xa[c]).categories
        for x in (xa, xb):
            codes = pd.Categorical(x[c], categories=levels).codes.astype(float)
            codes[codes < 0] = np.nan
            x[c] = codes
    model = HistGradientBoostingRegressor(max_iter=800, learning_rate=.08,
        max_leaf_nodes=127, categorical_features=list(range(4)), random_state=42)
    model.fit(xa, np.log(train.price))
    return np.exp(model.predict(xb))


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    frame = load_train()
    train, valid = time_split(frame, valid_from_year=2023)
    missing = pd.DataFrame({name: part.isna().sum() for name, part in
        [('train_before_2023', train), ('validation_2023', valid)]})
    missing.to_csv(OUT / 'missing_values.csv', index_label='column')
    corr = train[NUMERIC].corr()
    corr.to_csv(OUT / 'training_correlations.csv')
    for col in ['propertyType', 'tenure']:
        train[col].value_counts(dropna=False).to_csv(OUT / (col + '_training_counts.csv'))
    lower, upper = train.price.quantile([.005, .995])
    trimmed = train[train.price.between(lower, upper)]
    summary = {'training_rows':len(train), 'validation_rows':len(valid),
        'price_min':float(train.price.min()), 'price_max':float(train.price.max()),
        'trim_lower':float(lower), 'trim_upper':float(upper),
        'trimmed_training_rows':len(trimmed),
        'validation_policy':'Every experiment scores all 2023 rows; no validation trimming.',
        'scope':'Cleaning ablations use no previous-sale features. Validation is development evidence, not an untouched final test.'}
    (OUT / 'data_summary.json').write_text(json.dumps(summary, indent=2))
    jobs = [
        ('Median baseline', lambda:median_baseline(train, valid)),
        ('Original gradient boosting', lambda:gradient_boosting(train, valid)),
        ('Labels + missing flags', lambda:predict_clean(train, valid, False, False)),
        ('Labels + flags + rare tenure', lambda:predict_clean(train, valid, True, False)),
        ('Labels + flags + rare tenure + medians', lambda:predict_clean(train, valid, True, True)),
        ('Labels + flags + rare tenure + price trim', lambda:predict_clean(trimmed, valid, True, False)),
    ]
    rows = []
    for name, run in jobs:
        pred = run()
        mae = mean_absolute_error(valid.price, pred)
        rows.append({'experiment':name, 'mae_gbp':mae,
            'training_rows':len(trimmed) if name.endswith('price trim') else len(train),
            'validation_rows':len(valid)})
        print(f'{name}: GBP {mae:,.2f}', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'cleaning_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    values = missing.loc[missing.max(axis=1)>0, 'train_before_2023'].sort_values()
    values.plot.barh(ax=ax, color='#236a8d')
    ax.set(xlabel='Missing values in pre-2023 training rows', ylabel='', title='Training data completeness')
    fig.tight_layout();fig.savefig(OUT / 'figures/missing_values.png', dpi=150);plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(4), NUMERIC, rotation=35, ha='right');ax.set_yticks(range(4), NUMERIC)
    for i in range(4):
        for j in range(4): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',color='black')
    fig.colorbar(image,ax=ax);ax.set_title('Training feature correlations (Pearson)')
    fig.tight_layout();fig.savefig(OUT / 'figures/correlations.png',dpi=150);plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    results = pd.DataFrame(rows).sort_values('mae_gbp', ascending=False)
    ax.barh(results.experiment,results.mae_gbp,color='#236a8d')
    ax.set(xlabel='2023 validation MAE (GBP; lower is better)',title='Cleaning comparison: same model and validation rows')
    fig.tight_layout();fig.savefig(OUT / 'figures/cleaning_comparison.png',dpi=150);plt.close(fig)


if __name__ == '__main__':
    main()
