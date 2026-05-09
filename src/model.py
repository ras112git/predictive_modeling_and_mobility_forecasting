# Constants
DROP_COLS = ["id", "datetime", "name", "lat", "lng"]
TARGET = "bikes"

def prepare_features(df, drop_cols=DROP_COLS, target=TARGET):
    """Split a cleaned dataframe into (X, y). Returns (X, y)."""
    X = df.drop(columns=drop_cols + [target])
    y = df[target]
    return X, y


def feature_selection_report(X, y, sample_size=100_000, random_state=42):
    """Run correlation, mutual information, and RF permutation importance.
    Returns a DataFrame with one row per feature and columns
    {corr_max, mutual_info, perm_importance}, sorted by importance."""
    # Mark discrete columns
    from sklearn.feature_selection import mutual_info_regression

    X_sample = X.sample(n=sample_size, random_state=random_state)
    y_sample = y.loc[X_sample.index]

    discrete_cols = {'station_number', 'minute', 'dayofweek', 
                 'month', 'is_weekend', 'is_holiday', 'hour'}
    discrete_mask = [col in discrete_cols for col in X_sample.columns]

    mi = mutual_info_regression(
    X_sample, y_sample,
    discrete_features=discrete_mask,
    random_state=random_state)

    return mi

def get_model_grid(random_state=42):
    """Return a dict {name: (estimator, param_distribution)} for the benchmark.

    Each param_distribution is small (3-4 hyperparameters, narrow ranges) so a
    RandomizedSearchCV with ~5-10 iterations can cover a useful slice of it.

    All tree models use a Poisson objective/criterion to match the count
    nature of `bikes` (non-negative integers). This guarantees non-negative
    predictions and is more principled than squared error for count data.

    Imports for xgboost and lightgbm are lazy so the rest of this module
    keeps working even if those packages are not installed yet.
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.dummy import DummyRegressor
    return {
        "Featureless": (
            DummyRegressor(),
            {"strategy": ["mean", "median"]},
        ),

        "decision_tree": (
            DecisionTreeRegressor(criterion="poisson", random_state=random_state),
            {
                "max_depth": [10, 15, 20, None],
                "min_samples_leaf": [5, 10, 20, 50],
            },
        ),
        "random_forest": (
            RandomForestRegressor(
                criterion="poisson", n_jobs=-2, random_state=random_state
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 15, 25],
                "min_samples_leaf": [5, 10, 20],
                "max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "hist_gbm": (
            HistGradientBoostingRegressor(loss="poisson", random_state=random_state),
            {
                "max_iter": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [None, 8, 12],
                "min_samples_leaf": [10, 20, 50],
            },
        ),
        "xgboost": (
            XGBRegressor(
                objective="count:poisson",
                tree_method="hist",
                n_jobs=-2,
                random_state=random_state,
            ),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [6, 8, 10],
                "subsample": [0.8, 1.0],
            },
        ),
        "lightgbm": (
            LGBMRegressor(
                objective="poisson",
                n_jobs=-2,
                random_state=random_state,
                verbose=-1,
            ),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 63, 127],
                "min_child_samples": [10, 20, 50],
            },
        ),
    }


def get_final_param_grid(model_name):
    """Return a wider param distribution for the winning model,
    suitable for the final 30-50 iteration RandomizedSearchCV."""

def benchmark_models(
    X,
    y,
    models=None,
    outer_splits=3,
    inner_splits=3,
    n_iter=8,
    sample_size=None,
    time_aware=True,
    random_state=42,
    verbose=True,
):
    """Run nested CV over a grid of models and return a ranking DataFrame.

    Outer loop: TimeSeriesSplit (or KFold if `time_aware=False`) — gives an
    unbiased generalization estimate per model.
    Inner loop: RandomizedSearchCV with the same splitter — does a small
    hyperparameter search inside each outer training fold.

    Args:
        X, y: features and target. If time_aware=True, X must already be
            sorted by datetime (the datetime column itself is dropped earlier
            by prepare_features, so the order is what carries the time info).
        models: optional dict {name: (estimator, param_dist)}. Defaults to
            get_model_grid().
        outer_splits, inner_splits: number of CV folds.
        n_iter: iterations of RandomizedSearchCV inside each outer fold.
        sample_size: optional row subsample. Default None = use full data
            (recommended). Provide a smaller value only for development;
            rankings can shift at small sample sizes.
        time_aware: use TimeSeriesSplit.
        random_state: seed for the inner search and any subsampling.

    Returns:
        DataFrame sorted by mean_rmse with columns
        {model, mean_rmse, std_rmse, mean_mae, mean_poisson_deviance,
         best_params_per_fold}.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import (
        RandomizedSearchCV, TimeSeriesSplit, KFold,
    )
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, mean_poisson_deviance,
    )

    if models is None:
        models = get_model_grid(random_state=random_state)

    if sample_size is not None and sample_size < len(X):
        X = X.sample(n=sample_size, random_state=random_state).sort_index()
        y = y.loc[X.index]

    Splitter = TimeSeriesSplit if time_aware else KFold
    outer = Splitter(n_splits=outer_splits)
    inner = Splitter(n_splits=inner_splits)

    rows = []
    for name, (estimator, param_dist) in models.items():
        rmses, maes, devs, best_params = [], [], [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X)):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            search = RandomizedSearchCV(
                estimator,
                param_dist,
                n_iter=n_iter,
                cv=inner,
                scoring="neg_root_mean_squared_error",
                # estimators already use n_jobs=-1; avoid nested parallelism
                n_jobs=1,
                random_state=random_state,
                refit=True,
            )
            search.fit(X_tr, y_tr)

            # Poisson-trained models give non-negative preds; clip is defensive
            y_pred = np.maximum(0, search.predict(X_te))

            rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
            maes.append(mean_absolute_error(y_te, y_pred))
            # mean_poisson_deviance requires y_pred > 0
            devs.append(mean_poisson_deviance(y_te, np.maximum(1e-9, y_pred)))
            best_params.append(search.best_params_)

            if verbose:
                print(
                    f"  {name} fold {fold_idx + 1}/{outer_splits}: "
                    f"RMSE={rmses[-1]:.3f}  MAE={maes[-1]:.3f}"
                )

        rows.append({
            "model": name,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_mae": float(np.mean(maes)),
            "mean_poisson_deviance": float(np.mean(devs)),
            "rmse_per_fold": rmses,
            "best_params_per_fold": best_params,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )

def train_final_model(X, y, model_name, n_iter=40, cv_splits=3, random_state=42):
    """Run RandomizedSearchCV with a wider grid on the chosen model.
    Returns the fitted best_estimator_ and the search object."""

def evaluate(model, X, y):
    """Return a dict {rmse, mae, rmsle} for predictions on (X, y)."""

def save_model(model, path):
    """joblib.dump wrapper that creates parent directories."""

def plot_benchmark(df, metric="rmse_per_fold", figsize=(8, 5), title=None):
    """Box plot of per-fold scores from benchmark_models output.

    Args:
        df: DataFrame returned by benchmark_models.
        metric: column name holding a list of per-fold scores. Defaults to
            'rmse_per_fold'.
        figsize: matplotlib figure size.
        title: optional plot title.

    Returns:
        (fig, ax) tuple. Caller is responsible for fig.show() / savefig().
    """
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("mean_rmse", ascending=False)  # best at the top
    data = df_sorted[metric].tolist()
    labels = df_sorted["model"].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, vert=False, tick_labels=labels, showmeans=True)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(title or "Model benchmark — per-fold RMSE")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax

