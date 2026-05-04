# predictive_modeling_and_mobility_forecasting
Kaggle challenge competition, with the goal of training a machine learning agent in public transport data

# Objective
This project has the objective of training different agents for analyzing and predicting public transportation usage based on a given 6 month dataset

# Structure: Cookiecutter Data Science
predictive_modeling_and_mobility_forecasting/
│
├── README.md                  # Project description, setup instructions
├── .gitignore                 # Tells Git what to IGNORE
├── requirements.txt           # Python packages needed
│
├── notebooks/                 # Jupyter/Colab notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   └── 03_modeling.ipynb
│
├── src/                       # Reusable Python code (.py files)
│   ├── download_data.py
│   ├── data_cleaning.py
│   ├── features.py
│   └── model.py
│
├── data/
│   ├── raw/                   # Original, untouched data (NOT in Git)
│   ├── interim/               # Partially cleaned data (NOT in Git)
│   └── processed/             # Final clean data (NOT in Git)
│
├── models/                    # Saved trained models (usually NOT in Git)
│
└── reports/
    └── figures/               # Plots and visualizations

# Data cleaning pipeline

Google Drive (raw data, shared)
        ↓
   download_data.py   ← fetches into data/raw/
        ↓
   data/raw/dataset_train.csv, data/raw/dataset_test.csv   (gitignored, exist locally only)
        ↓
   clean_data.py     ← transforms raw into clean
        ↓
   data/processed/dataset_clean.csv   (gitignored)
        ↓
   notebooks use the clean version

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the data preparation pipeline: `python scripts/prepare_data.py`

This downloads the raw data from our shared Google Drive into `data/raw/`
and produces a cleaned version in `data/processed/`.

## Data description

The data correspond to thousands of bike trips for the nextbike-system in Vienna. It only contains entries for station-interval combinations for which data was observed.

- `dataset_train.csv` - the training set, containing data from `2024-09-03 17:30:00` to `2025-03-06 10:00:00`

- `dataset_test.csv` - the test set, containing data from `2025-03-06 10:30:00` to `2025-04-29 23:30:00`
