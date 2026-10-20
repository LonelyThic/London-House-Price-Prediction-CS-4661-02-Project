# London House Price Prediction

CS4661 – Introduction to Data Science · Group Project

Predicting London property sale prices with regression models, based on the Kaggle competition
[London House Price Prediction: Advanced Techniques](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques).
Models are evaluated on **Mean Absolute Error (MAE)**.

## Team

| Name | Role / Responsibilities |
|------|-------------------------|
|      |                         |
|      |                         |
|      |                         |

## Project Structure

```
london-house-prices/
├── data/              # Kaggle CSVs go here (not committed)
├── notebooks/         # Exploration and experiments, one notebook per person/task
├── src/               # Shared Python code (loading, cleaning, features, models)
├── reports/figures/   # Plots for the report and slides
├── submissions/       # Prediction files (not committed)
├── requirements.txt
└── README.md
```

## Setup

Run these once after cloning:

```bash
git clone <repo-url>
cd london-house-prices
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate #skip if got error for now
pip install -r requirements.txt
nbstripout --install             # strips notebook outputs before each commit
```

In VS Code, select the `.venv` interpreter with **Ctrl/Cmd+Shift+P → Python: Select Interpreter**.

## Getting the Data

The data is **not** stored in this repo. Get the files in one of two ways:

- Unzip the zip the team shared into `data/`.
- Or download it with the Kaggle CLI (you must accept the competition rules on Kaggle first):

```bash
  kaggle competitions download -c london-house-price-prediction-advanced-techniques -p data
  unzip data/london-house-price-prediction-advanced-techniques.zip -d data
```

Either way, you should end up with:

```
data/train.csv
data/test.csv
data/sample_submission.csv
```

| File | Rows | Notes |
|------|------|-------|
| `train.csv` | 266,325 | Sales from 1995–2023, includes `price` |
| `test.csv` | 16,547 | Sales from 2024, no `price` |

## Using Shared Code in Notebooks

```python
import sys; sys.path.append("..")
from src.data import load_train, load_test, time_split
```

## Validation Approach

The test set contains only 2024 sales, so we validate on the most recent years rather than with a random split.
For example, train on sales before 2022 and validate on 2022–2023.
This gives a realistic estimate of how the model will do on future sales.

## Team Workflow

1. Pull the latest `main` before starting: `git pull origin main`
2. Create your own branch: `git checkout -b <task>-<name>` (e.g. `eda-maria`)
3. Commit often with clear messages.
4. Push and open a pull request into `main`. At least one teammate reviews it before merging.
5. Put reusable code in `src/` rather than copying it between notebooks.
6. Name notebooks `NN_topic_name.ipynb` (e.g. `02_cleaning_alex.ipynb`) so two people never edit the same one.

## Academic Integrity

All code, analysis, and writing in this repo is the team's original work.
We do not copy code from Kaggle notebooks or other public solutions.

## Deliverables

- [ ] Project report (methods, algorithms, results, each member's responsibilities)
- [ ] PowerPoint slides
- [ ] Final code in this repo