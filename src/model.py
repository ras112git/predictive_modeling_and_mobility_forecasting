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


def get_bayes_search_space(Feature_df, model_name):
    """Return continuous/integer search-space specs for Bayesian optimization.

    The Optuna counterpart of `get_final_param_grid`: instead of discrete value
    lists it returns, per hyperparameter, a distribution spec that
    `train_hyperparameters` feeds to `optuna.Trial.suggest_*`. Encoding:

        ("int",   low, high)            -> trial.suggest_int(name, low, high)
        ("int",   low, high, "log")     -> trial.suggest_int(..., log=True)
        ("float", low, high)            -> trial.suggest_float(name, low, high)
        ("float", low, high, "log")     -> trial.suggest_float(..., log=True)
        ("cat",   [values])             -> trial.suggest_categorical(name, values)

    Ranges are roughly the convex hull of the discrete grids in
    `get_final_param_grid`, so Bayesian search can exploit continuity rather
    than hopping between a handful of fixed points.

    Args:
        Feature_df: the feature frame — only inspected for `station_number`,
            which decides whether the pipeline `model__` prefix is applied
            (mirrors `get_final_param_grid`).
        model_name: key from `get_model_grid()`.

    Returns:
        Dict {param_name: spec_tuple} suitable for `train_hyperparameters`.

    Raises:
        KeyError: if `model_name` is not one of the known models.
    """
    spaces = {
        "Featureless": {
            "strategy": ("cat", ["mean", "median"]),
        },
        "decision_tree": {
            "max_depth":         ("int",   4, 32),
            "min_samples_leaf":  ("int",   1, 100, "log"),
            "min_samples_split": ("int",   2, 20),
            "max_features":      ("float", 0.3, 1.0),
        },
        "random_forest": {
            "n_estimators":      ("int",   200, 600),
            "max_depth":         ("int",   8, 32),
            "min_samples_leaf":  ("int",   1, 10),
            "min_samples_split": ("int",   2, 10),
            "max_features":      ("float", 0.3, 0.8),
        },
        "hist_gbm": {
            "max_iter":          ("int",   200, 1200),
            "learning_rate":     ("float", 0.01, 0.2, "log"),
            "max_depth":         ("int",   4, 16),
            "min_samples_leaf":  ("int",   10, 100),
            "max_leaf_nodes":    ("int",   15, 255),
            "l2_regularization": ("float", 1e-3, 10.0, "log"),
        },
        "xgboost": {
            "n_estimators":      ("int",   200, 1200),
            "learning_rate":     ("float", 0.005, 0.2, "log"),
            "max_depth":         ("int",   4, 12),
            "min_child_weight":  ("int",   1, 20),
            "subsample":         ("float", 0.6, 1.0),
            "colsample_bytree":  ("float", 0.6, 1.0),
            "reg_alpha":         ("float", 1e-3, 1.0, "log"),
            "reg_lambda":        ("float", 1e-2, 10.0, "log"),
            "gamma":             ("float", 1e-3, 1.0, "log"),
        },
        "lightgbm": {
            "n_estimators":      ("int",   200, 1200),
            "learning_rate":     ("float", 0.005, 0.2, "log"),
            "num_leaves":        ("int",   16, 512),
            "max_depth":         ("int",   4, 14),
            "min_child_samples": ("int",   5, 100),
            "subsample":         ("float", 0.6, 1.0),
            "colsample_bytree":  ("float", 0.6, 1.0),
            "reg_alpha":         ("float", 1e-3, 1.0, "log"),
            "reg_lambda":        ("float", 1e-2, 10.0, "log"),
        },
    }
    if model_name not in spaces:
        raise KeyError(
            f"Unknown model '{model_name}'. Expected one of {list(spaces)}."
        )

    space = spaces[model_name]

    # Pipeline-wrapped models need the model__ prefix (same rule as the grid).
    has_station = 'station_number' in Feature_df.columns
    if has_station and model_name in {"decision_tree", "random_forest"}:
        space = {f"model__{k}": v for k, v in space.items()}
    return space


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

def _build_final_estimator(X, model_name, random_state=42, log_target=False):
    """Build the (optionally log-transformed) estimator + its param grid.

    Shared by `train_hyperparameters` and `train_final_model` so both
    construct the estimator identically. That identity is what makes the
    `regressor__`-prefixed `best_params` returned by the search applicable,
    unchanged, in the final refit.

    When `log_target=True`, the estimator's loss is switched from Poisson to
    squared error and the whole thing is wrapped in
    TransformedTargetRegressor(func=log1p, inverse_func=expm1).

    Returns:
        (estimator, param_dist) where `param_dist` already has the
        `regressor__` prefix applied when `log_target=True`.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor
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

    return estimator, param_dist


def train_hyperparameters(
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
    """Bayesian (Optuna/TPE) hyperparameter search; returns the best params.

    Runs a TimeSeriesSplit CV for each Optuna trial and minimizes RMSE (or
    RMSLE when `log_target=True`, so hyperparameters are picked on the metric
    Kaggle scores you on). Unlike a random sweep, each trial is chosen from the
    results of earlier ones, over the continuous/integer ranges defined in
    `get_bayes_search_space`. Does NOT refit a final model — hand the returned
    `best_params` to `train_final_model`.

    Args:
        n_iter: number of Optuna trials (kept as `n_iter` for call-site
            compatibility with the previous random search).
        sample_size: if set, the search uses only the last `sample_size` rows
            (the most recent data) to speed up tuning. Independent of the
            sample size used by `train_final_model` for the final fit.

    Returns:
        (best_params, search) where `best_params` is the winning param dict
        (keys carry the `regressor__` prefix when `log_target=True`, ready to
        pass straight to `train_final_model`), and `search` is a
        SimpleNamespace with `best_params_`, `best_score_`, `results_df`.
    """
    import time
    from types import SimpleNamespace

    import numpy as np
    import optuna
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error, mean_squared_log_error
    from sklearn.model_selection import TimeSeriesSplit

    # Reuse the shared builder so the estimator is constructed identically to
    # train_final_model (loss-switch + TransformedTargetRegressor under
    # log_target). Its param_dist is the random-search grid — not needed here.
    estimator, _ = _build_final_estimator(
        X, model_name, random_state=random_state, log_target=log_target
    )

    space = get_bayes_search_space(X, model_name)
    if log_target:
        # Route every param through the TransformedTargetRegressor wrapper.
        space = {f"regressor__{k}": v for k, v in space.items()}

    if sample_size is not None and sample_size < len(X):
        X = X.iloc[-sample_size:]
        y = y.iloc[-sample_size:]

    cv = TimeSeriesSplit(n_splits=cv_splits)
    metric_name = "rmsle" if log_target else "rmse"

    def _suggest(trial, name, spec):
        kind = spec[0]
        if kind == "cat":
            return trial.suggest_categorical(name, spec[1])
        low, high = spec[1], spec[2]
        log = len(spec) > 3 and spec[3] == "log"
        if kind == "int":
            return trial.suggest_int(name, low, high, log=log)
        if kind == "float":
            return trial.suggest_float(name, low, high, log=log)
        raise ValueError(f"Unknown spec kind '{kind}' for param '{name}'.")

    if verbose:
        print(
            f"[train_hyperparameters] model={model_name}  log_target={log_target}  "
            f"rows={len(X):,}  features={X.shape[1]}  n_iter={n_iter}  "
            f"cv_splits={cv_splits}  total_fits={n_iter * cv_splits}  "
            f"sampler=TPE  scoring={metric_name}"
        )
        print(f"[train_hyperparameters] search space keys: {sorted(space)}\n")

    def objective(trial):
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
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
        trial.set_user_attr("fold_scores", fold_scores)
        return float(np.mean(fold_scores))

    def _progress(study, trial):
        if not verbose:
            return
        fold_scores = trial.user_attrs.get("fold_scores", [])
        print(
            f"[{trial.number + 1:>3}/{n_iter}] mean_{metric_name}={trial.value:.4f}  "
            f"std={float(np.std(fold_scores)):.4f}  "
            f"folds={[f'{r:.3f}' for r in fold_scores]}  "
            f"best={study.best_value:.4f}  params={trial.params}",
            flush=True,
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    t_start = time.time()
    study.optimize(objective, n_trials=n_iter, callbacks=[_progress])

    rows = [
        {
            "iter": t.number + 1,
            f"mean_{metric_name}": t.value,
            f"std_{metric_name}": float(np.std(t.user_attrs.get("fold_scores", [0.0]))),
            "params": t.params,
            f"fold_{metric_name}s": t.user_attrs.get("fold_scores", []),
        }
        for t in study.trials
    ]
    results_df = (
        pd.DataFrame(rows)
        .sort_values(f"mean_{metric_name}")
        .reset_index(drop=True)
    )
    best_params = study.best_params

    if verbose:
        print(
            f"\n[train_hyperparameters] search done in {time.time() - t_start:.1f}s. "
            f"Best mean_{metric_name}={study.best_value:.4f}  params={best_params}"
        )

    search = SimpleNamespace(
        best_params_=best_params,
        best_score_=-study.best_value,
        results_df=results_df,
    )
    return best_params, search


def train_final_model(
    X,
    y,
    model_name,
    best_params,
    sample_size=None,
    random_state=42,
    log_target=False,
    verbose=True,
):
    """Refit the chosen model with given hyperparameters on the data.

    Args:
        best_params: param dict from `train_hyperparameters`. Its keys must
            carry the same prefixes used during tuning (e.g. `regressor__`
            when `log_target=True`), and `log_target` here must match what was
            passed to `train_hyperparameters` so the estimator is rebuilt the
            same way.
        sample_size: if set, fit on only the last `sample_size` rows (the most
            recent data). Default None = use all of (X, y).

    Returns:
        The fitted estimator.
    """
    from sklearn.base import clone

    estimator, _ = _build_final_estimator(
        X, model_name, random_state=random_state, log_target=log_target
    )

    if sample_size is not None and sample_size < len(X):
        X = X.iloc[-sample_size:]
        y = y.iloc[-sample_size:]

    if verbose:
        print(
            f"[train_final_model] refitting {model_name} on "
            f"rows={len(X):,}, features={X.shape[1]}, log_target={log_target}..."
        )
        print(f"[train_final_model] params: {best_params}")

    best_estimator = clone(estimator).set_params(**best_params)
    best_estimator.fit(X, y)
    return best_estimator

def evaluate(model, X, y):
    """Return a dict {rmse, mae, rmsle} for predictions on (X, y)."""

    from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_squared_log_error
    import numpy as np

    # Poisson/log-target models can emit tiny negatives; clip so the metrics
    # (RMSLE in particular) match what the rounded, non-negative submission scores.
    y_pred = np.maximum(0, model.predict(X))

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
    print(f"MAE:   {mae:.3f}")
    print(f"RMSE:  {rmse:.3f}")
    print(f"RMSLE: {rmsle:.3f}")

    return {"mae": mae, "rmse": rmse, "rmsle": rmsle}

def save_model(model, path):
    """joblib.dump wrapper that creates parent directories."""
    import os
    import joblib

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    return path

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