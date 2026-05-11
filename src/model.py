def data_split(df, target):
    """Split a cleaned dataframe into (X, y). Returns (X, y)."""
    X = df.drop(columns = target)
    y = df[target]
    return X, y

def prepare_features(X, drop_cols):
    """Drop identifier columns and cast station_number to pandas `category`.

    The categorical dtype is what LightGBM, XGBoost (with enable_categorical)
    and HistGradientBoosting auto-detect as a categorical feature, so the
    modeling code doesn't need a separate `categorical_feature` argument.
    """
    X = X.drop(columns=drop_cols)
    if 'station_number' in X.columns:
        X = X.assign(station_number=X['station_number'].astype('category'))
    return X


def feature_selection_report(X, y, sample_size=100_000, random_state=42):
    """Run correlation, mutual information, and RF permutation importance.
    Returns a DataFrame with one row per feature and columns
    {corr_max, mutual_info, perm_importance}, sorted by importance."""
    # Mark discrete columns
    from sklearn.feature_selection import mutual_info_regression

    X_sample = X.sample(n=sample_size, random_state=random_state)
    y_sample = y.loc[X_sample.index]

    has_station = 'station_number' in X.columns

    if has_station:
        discrete_cols = {'station_number','minute', 'dayofweek', 
                    'month', 'is_weekend', 'is_holiday', 'hour'}
    else:
        discrete_cols = {'minute', 'dayofweek', 
            'month', 'is_weekend', 'is_holiday', 'hour'} 
    
    discrete_mask = [col in discrete_cols or col.startswith('st_') for col in X_sample.columns]

    mi = mutual_info_regression(
    X_sample, y_sample,
    discrete_features=discrete_mask,
    random_state=random_state)

    return mi

def get_model_grid(Feature_df, random_state=42):
    """Return a dict {name: (estimator, param_distribution)} for the benchmark.

    Tree models use a Poisson objective/criterion for the count target.
    `station_number` is handled natively by hist_gbm/xgboost/lightgbm, and
    target-encoded (CV-safe, mean bikes per station with empirical-Bayes
    shrinkage) for decision_tree and random_forest, which don't support
    categorical splits.
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.preprocessing import TargetEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.dummy import DummyRegressor

    has_station = 'station_number' in Feature_df.columns

    # --- Native-categorical model params ---
    xgb_params = dict(
        objective="count:poisson",
        tree_method="hist",
        n_jobs=-2,
        random_state=random_state,
    )
    lgbm_params = dict(
        objective="poisson",
        n_jobs=-2,
        random_state=random_state,
        verbose=-1,
    )
    hgb_params = dict(loss="poisson", random_state=random_state)

    if has_station:
        # All three rely on pandas `category` dtype on station_number, which
        # prepare_features() sets. LightGBM auto-detects category dtype, so no
        # explicit categorical_feature argument is needed (and passing one in
        # the constructor dict triggers a warning + gets overridden).
        xgb_params["enable_categorical"] = True
        hgb_params["categorical_features"] = ['station_number']

    # --- Helper: wrap an estimator so station_number gets target-encoded
    #     and everything else is passed through. ColumnTransformer is
    #     re-fit on each CV split, so the encoding is leak-free.
    def wrap_with_te(estimator):
        if not has_station:
            return estimator
        preprocessor = ColumnTransformer(
            transformers=[
                ("station_te", TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    random_state=random_state,
                ), ["station_number"]),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        return Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator),
        ])

    # When wrapped in a Pipeline, hyperparameters need a "model__" prefix
    # so RandomizedSearchCV / set_params() route them to the inner step.
    prefix = "model__" if has_station else ""

    return {
        "Featureless": (
            DummyRegressor(),
            {"strategy": ["mean", "median"]},
        ),
        "decision_tree": (
            wrap_with_te(DecisionTreeRegressor(
                criterion="poisson", random_state=random_state,
            )),
            {
                f"{prefix}max_depth": [10, 15, 20, None],
                f"{prefix}min_samples_leaf": [5, 10, 20, 50],
            },
        ),
        "random_forest": (
            wrap_with_te(RandomForestRegressor(
                criterion="poisson", n_jobs=-2, random_state=random_state,
            )),
            {
                f"{prefix}n_estimators": [100, 200],
                f"{prefix}max_depth": [None, 15, 25],
                f"{prefix}min_samples_leaf": [5, 10, 20],
                f"{prefix}max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "hist_gbm": (
            HistGradientBoostingRegressor(**hgb_params),
            {
                "max_iter": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [None, 8, 12],
                "min_samples_leaf": [10, 20, 50],
            },
        ),
        "xgboost": (
            XGBRegressor(**xgb_params),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [6, 8, 10],
                "subsample": [0.8, 1.0],
            },
        ),
        "lightgbm": (
            LGBMRegressor(**lgbm_params),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 63, 127],
                "min_child_samples": [10, 20, 50],
            },
        ),
    }

def get_final_param_grid(Feature_df, model_name):
    """Return a wider param distribution for the winning model.

    Sized for a final RandomizedSearchCV with ~40 iterations: each grid
    covers 5-9 hyperparameters with ranges roughly 2-3x wider than the
    benchmark grid in `get_model_grid`. Param names are model-specific
    (e.g. `num_leaves` only for LightGBM, `max_features` only for tree/RF).

    Args:
        model_name: key from `get_model_grid()` — also matches the values
            in the `model` column of the benchmark ranking DataFrame.

    Returns:
        Dict suitable as the `param_distributions` argument of
        RandomizedSearchCV.

    Raises:
        KeyError: if `model_name` is not one of the known models.
    """
    grids = {
        "Featureless": {
            "strategy": ["mean", "median"],
        },
        "decision_tree": {
            "max_depth": [None, 8, 12, 15, 20, 25, 30],
            "min_samples_leaf": [1, 5, 10, 20, 50, 100],
            "min_samples_split": [2, 5, 10, 20],
            "max_features": [None, "sqrt", 0.5, 0.7, 1.0],
        },
        "random_forest": {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 15, 20, 25, 30],
            "min_samples_leaf": [1, 5, 10],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", 0.3, 0.5, 0.7],
        },
        "hist_gbm": {
            "max_iter": [200, 400, 600, 800, 1000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "max_depth": [None, 6, 8, 10, 12, 15],
            "min_samples_leaf": [10, 20, 50, 100],
            "max_leaf_nodes": [15, 31, 63, 127],
            "l2_regularization": [0.0, 0.1, 1.0, 10.0],
        },
        "xgboost": {
            "n_estimators": [200, 400, 600, 800, 1000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "max_depth": [4, 6, 8, 10, 12],
            "min_child_weight": [1, 5, 10, 20],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.1, 1.0, 10.0],
            "gamma": [0, 0.1, 0.5, 1.0],
        },
        "lightgbm": {
            "n_estimators": [200, 400, 600, 800, 1000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "num_leaves": [31, 63, 127, 255, 511],
            "max_depth": [-1, 6, 8, 10, 12],
            "min_child_samples": [5, 10, 20, 50, 100],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.1, 1.0, 10.0],
        },
    }
    if model_name not in grids:
        raise KeyError(
            f"Unknown model '{model_name}'. Expected one of {list(grids)}."
        )
    
    grid = grids[model_name]

    # Pipeline-wrapped models need the model__ prefix
    has_station = 'station_number' in Feature_df.columns

    if has_station and model_name in {"decision_tree", "random_forest"}:
        grid = {f"model__{k}": v for k, v in grid.items()}
    return grid


def benchmark_models(
    X,
    y,
    models=None,
    outer_splits=3,
    inner_splits=3,
    n_iter=8,
    sample_size=200_000,
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
        mean_squared_error, mean_absolute_error, mean_poisson_deviance,mean_squared_log_error
    )

    if models is None:
        models = get_model_grid(Feature_df = X, random_state=random_state)

    if sample_size is not None and sample_size < len(X):
        X = X.sample(n=sample_size, random_state=random_state).sort_index()
        y = y.loc[X.index]

    Splitter = TimeSeriesSplit if time_aware else KFold
    outer = Splitter(n_splits=outer_splits)
    inner = Splitter(n_splits=inner_splits)

    rows = []
    for name, (estimator, param_dist) in models.items():
        rmses, maes, rmsles, devs, best_params = [], [], [], [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X)):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            search = RandomizedSearchCV(
                estimator,
                param_dist,
                n_iter=n_iter,
                cv=inner,
                scoring="neg_root_mean_squared_error",
                # estimators use n_jobs=-2; Do not use more than n_jobs=1, as it would cause deadlock
                n_jobs=1,
                random_state=random_state,
                refit=True,
            )
            search.fit(X_tr, y_tr)

            # Poisson-trained models give non-negative preds; clip is defensive
            y_pred = np.maximum(0, search.predict(X_te))

            rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
            maes.append(mean_absolute_error(y_te, y_pred))
            rmsles.append(np.sqrt(mean_squared_log_error(y_te, y_pred)))
            # mean_poisson_deviance requires y_pred > 0
            devs.append(mean_poisson_deviance(y_te, np.maximum(1e-9, y_pred)))
            best_params.append(search.best_params_)

            if verbose:
                print(
                    f"  {name} fold {fold_idx + 1}/{outer_splits}: "
                    f"RMSE={rmses[-1]:.3f}  MAE={maes[-1]:.3f} RMSLE={rmsles[-1]:.3f}"
                )

        rows.append({
            "model": name,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_mae": float(np.mean(maes)),
            "mean_rmsle": float(np.mean(rmsles)),
            "std_rmsle": float(np.std(rmsles)),
            "mean_poisson_deviance": float(np.mean(devs)),
            "rmse_per_fold": rmses,
            "best_params_per_fold": best_params,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )

def train_final_model(
    X,
    y,
    model_name,
    n_iter=40,
    cv_splits=3,
    random_state=42,
    verbose=True,
    sample_size=None,
    log_target=False,
):
    """Run a manual randomized search with a wider grid on the chosen model.

    When `log_target=True`, the estimator's loss is switched from Poisson
    to squared error and the whole thing is wrapped in
    TransformedTargetRegressor(func=log1p, inverse_func=expm1). CV scoring
    also switches from RMSE to RMSLE so hyperparameters are picked on the
    metric Kaggle scores you on.
    """
    import time
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.metrics import mean_squared_error, mean_squared_log_error
    from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
    from sklearn.pipeline import Pipeline

    estimator, _ = get_model_grid(Feature_df=X, random_state=random_state)[model_name]
    param_dist = get_final_param_grid(Feature_df=X, model_name=model_name)

    if log_target:
        # Map each model to the param that controls its loss function.
        loss_settings = {
            "decision_tree": ("criterion", "squared_error"),
            "random_forest": ("criterion", "squared_error"),
            "hist_gbm":      ("loss",      "squared_error"),
            "xgboost":       ("objective", "reg:squarederror"),
            "lightgbm":      ("objective", "regression"),
        }
        if model_name in loss_settings:
            pname, pval = loss_settings[model_name]
            # DT/RF come back as Pipeline([preprocess, model]) when station_number
            # is present, so the param is reachable as "model__criterion", this comes 
            # from the pipeline for decision trees and random forests
            if isinstance(estimator, Pipeline):
                pname = f"model__{pname}"
            estimator = clone(estimator).set_params(**{pname: pval})

        estimator = TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
        # Every existing key needs the regressor__ prefix to route through the wrapper.
        param_dist = {f"regressor__{k}": v for k, v in param_dist.items()}

    if sample_size is not None and sample_size < len(X):
        X_total = X 
        y_total = y
        X = X.iloc[-sample_size:]
        y = y.iloc[-sample_size:]

    cv = TimeSeriesSplit(n_splits=cv_splits)
    sampler = list(
        ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state)
    )

    metric_name = "rmsle" if log_target else "rmse"
    if verbose:
        print(
            f"[train_final_model] model={model_name}  log_target={log_target}  "
            f"rows={len(X):,}  features={X.shape[1]}  n_iter={len(sampler)}  "
            f"cv_splits={cv_splits}  total_fits={len(sampler) * cv_splits + 1} "
            f"(incl. final refit)  scoring={metric_name}"
        )
        print(f"[train_final_model] param grid keys: {sorted(param_dist)}\n")

    rows = []
    t_start = time.time()
    for i, params in enumerate(sampler, start=1):
        t_cfg = time.time()
        fold_scores = []
        for tr_idx, te_idx in cv.split(X):
            est = clone(estimator).set_params(**params)
            est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            y_pred = np.maximum(0, est.predict(X.iloc[te_idx]))
            if log_target:
                fold_scores.append(
                    float(np.sqrt(mean_squared_log_error(y.iloc[te_idx], y_pred)))
                )
            else:
                fold_scores.append(
                    float(np.sqrt(mean_squared_error(y.iloc[te_idx], y_pred)))
                )
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        rows.append({
            "iter": i,
            f"mean_{metric_name}": mean_score,
            f"std_{metric_name}": std_score,
            "params": params,
            f"fold_{metric_name}s": fold_scores,
        })
        if verbose:
            print(
                f"[{i:>3}/{len(sampler)}] mean_{metric_name}={mean_score:.4f}  "
                f"std={std_score:.4f}  folds={[f'{r:.3f}' for r in fold_scores]}  "
                f"time={time.time() - t_cfg:.1f}s  params={params}",
                flush=True,
            )

    results_df = (
        pd.DataFrame(rows)
        .sort_values(f"mean_{metric_name}")
        .reset_index(drop=True)
    )
    best_row = results_df.iloc[0]
    best_params = best_row["params"]

    if verbose:
        print(
            f"\n[train_final_model] search done in {time.time() - t_start:.1f}s. "
            f"Best mean_{metric_name}={best_row[f'mean_{metric_name}']:.4f}  "
            f"params={best_params}"
        )
        print("[train_final_model] refitting best config on full (X, y)...")

    best_estimator = clone(estimator).set_params(**best_params)
    
    # train the last estimator with the complete dataset
    best_estimator.fit(X_total, y_total)

    search = SimpleNamespace(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=-best_row[f"mean_{metric_name}"],
        results_df=results_df,
    )
    return best_estimator, search

def evaluate(model, X, y):
    """Return a dict {rmse, mae, rmsle} for predictions on (X, y)."""

    from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_squared_log_error
    import numpy as np

    y_pred = model.predict(X)

    print(y_pred)

    # Evaluate
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
    print(f"MAE:   {mae:.3f}")
    print(f"RMSE:  {rmse:.3f}")
    print(f"RMSLE: {rmsle:.3f}")

    return None

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

