import os
import pandas as pd
import numpy as np
import holidays



def clean_data(dataset, is_train: bool, categorize_station = True, station_categories=None):
    """
    Does the dataset cleaning that it was specified in the cleaning.ipynb

    Args:
    dataset: Which dataset wants to be cleaned
    is_train: True for training data (drops NA + duplicates), False for test
        (preserves rows for submission).
    categorize_station: True implies that the stations should be treated as categories, 
        false, the categories are splited in hot boolean columns (station_categories), it was seen
        that it does not make a difference, so there is now just a pipeline in the benchmark grid 
    station_categories: optional list of station numbers to use as one-hot
        categories. When cleaning the train set, leave as None — the unique
        stations in the data will be used. When cleaning the test set, pass
        the train-set station list so train and test share the same `st_*`
        columns in the same order. Stations in test but not in this list
        produce all-zero rows across the station columns.

    Returns:
    Cleaned dataset with `station_number` replaced by one-hot columns
    `st_<id>` (one per station in `station_categories`).
    """
    dataset = dataset.copy()
    
    #Transform the datetime, needed for the submission file, the localize makes sure that the format is correct (rather than having the UCT reference)
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], utc=True).dt.tz_localize(None)  
    
    # This is the old factor changing method
    """
    # Boolean check, that it is true when hour, minute and second are all 0
    is_midnight = (dataset['datetime'].dt.hour == 0) & (dataset['datetime'].dt.minute == 0) & (dataset['datetime'].dt.second == 0)
    
    # What where does is, that when it is midnight, then it 
    datetime_str = np.where(
        is_midnight,
        dataset['datetime'].dt.strftime('%Y-%m-%d'),
        dataset['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    """
    datetime_str = dataset['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # The insert method takes (position_index, column_name, values)
    dataset.insert(0, 'id', datetime_str + "_" + dataset.iloc[:, 1].astype(str))

    if is_train:    
        # If there are rows with NA, then drop them
        if dataset.isna().any(axis=1).any():
            dataset = dataset.dropna()

        # If there are duplicates, then drop them.
        if dataset.duplicated().any():
            dataset = dataset.drop_duplicates()

    #I want to split my datainto the datapart (how already fastai tabular uses to learn)
    dataset['hour'] = dataset['datetime'].dt.hour
    dataset['minute'] = dataset['datetime'].dt.minute  # 0 or 30
    dataset['dayofweek'] = dataset['datetime'].dt.dayofweek
    dataset['month'] = dataset['datetime'].dt.month
    dataset['is_weekend'] = dataset['dayofweek'].isin([5, 6]).astype(int)

    # For cyclical data I can help the model differenciate between simple ordered rankings and actual circular data. In this case I do it with the hour (so it does not jump from 23 to 0 directly)
    dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24)
    dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24)

    # Create Austria holidays object
    at_holidays = holidays.Austria(subdiv='9') #subdiv = 9 locates the specific holidays of viena


    # Create the holiday column, also transform it into discrete
    dataset['is_holiday'] = dataset['datetime'].dt.date.apply(lambda x: x in at_holidays)
    dataset['is_holiday'] = dataset['is_holiday'].astype(int)

    # Average bikes per station (target-derived): on train this computes the
    # per-station means and caches them; on test it reads that cache back.
    # Done while station_number is still a plain int, before encoding below.
    dataset = add_avg_bikes_per_station(dataset, is_train=is_train)

    # One-hot encode station_number. Using pd.Categorical with explicit
    # categories guarantees that train and test produce the same st_* columns
    # in the same order when the caller passes station_categories.
    if categorize_station:
        dataset['station_number'] = dataset['station_number'].astype('category')
        
    else:
        if station_categories is None:
            station_categories = sorted(dataset['station_number'].unique())
        dataset['station_number'] = pd.Categorical(
            dataset['station_number'], categories=station_categories
        )
        dataset = pd.get_dummies(
            dataset, columns=['station_number'], prefix='st', dtype=int
        ) 


    dataset = add_weather_features(dataset)


    return dataset


def add_avg_bikes_per_station(
    df,
    is_train: bool,
    target="bikes",
    cache_path="data/interim/station_means.csv",
):
    """Append the mean number of bikes per station as a feature column.

    Some stations are consistently busier than others, so the per-station
    average of the target is a strong signal. But because it is derived from
    the target, it must be computed on the TRAIN data only and then reused
    unchanged on the test set — otherwise it leaks target information.

    Train and test are cleaned in separate notebook runs (03 vs 04), so the
    means are persisted to `cache_path`, the same train/test contract that
    `add_weather_features` uses for its weather cache: on the train pass the
    means are computed from `df` and written out; on the test pass they are
    read back and mapped onto the rows by `station_number`.

    Args:
        df: cleaned DataFrame with a `station_number` column. On the train
            pass it must also contain `target`.
        is_train: True computes the means from `df` and writes them to
            `cache_path`. False reads the means from `cache_path` and maps
            them on (no target column needed).
        target: name of the target column. Default `'bikes'`.
        cache_path: CSV path for the persisted per-station means. Pass None to
            skip persistence (only meaningful on the train pass, e.g. for an
            in-memory experiment).

    Returns:
        Copy of `df` with a new `avg_bikes_per_station` column. Stations not
        present in the means (e.g. a test station unseen in train) fall back
        to the global mean of the train means.

    Side effects:
        Writes the per-station means to `cache_path` on the train pass.

    Raises:
        KeyError: if `station_number` is missing, or `target` is missing on
            the train pass.
        FileNotFoundError: on the test pass if `cache_path` does not exist
            (clean the train set first to create it).
    """
    from pathlib import Path

    if "station_number" not in df.columns:
        raise KeyError(
            "df must have a 'station_number' column before averaging bikes "
            "per station."
        )

    df = df.copy()
    cache_path = Path(cache_path) if cache_path else None

    if is_train:
        if target not in df.columns:
            raise KeyError(
                f"target column '{target}' not found; computing the per-station "
                "means requires the training data."
            )
        station_means = df.groupby("station_number", observed=True)[target].mean()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            station_means.rename("avg_bikes_per_station").to_csv(cache_path)
    else:
        if cache_path is None or not cache_path.exists():
            raise FileNotFoundError(
                f"station means cache not found at {cache_path}. Clean the "
                "train set first (is_train=True) to create it."
            )
        cached = pd.read_csv(cache_path)
        station_means = cached.set_index("station_number")["avg_bikes_per_station"]

    # Global mean is the fallback for stations missing from the means mapping.
    global_mean = float(station_means.mean())
    df["avg_bikes_per_station"] = (
        df["station_number"].map(station_means).fillna(global_mean)
    )

    return df


def add_weather_features(
    df,
    lat: float = 48.21,
    lng: float = 16.37,
    cache_path="data/interim/weather_vienna.csv",
    variables=(
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "snowfall",
        "wind_speed_10m",
        "cloud_cover",
        "relative_humidity_2m",
    ),
    timeout: int = 60,
):
    """Append hourly weather columns to df from Open-Meteo's archive API.

    The date range is inferred from `df['datetime']`. Times are treated as
    UTC throughout to match the convention used by `clean_data`, which
    strips the tz after parsing as UTC. Each row is joined to the weather
    record for its hour-floored timestamp, so 30-minute rows (`:00` and
    `:30`) share the same weather row.

    Args:
        df: DataFrame with a 'datetime' column (naive datetime in UTC).
        lat, lng: location for the weather query. Default is central Vienna.
        cache_path: CSV path used to cache the API response. Reused when it
            covers the requested date range and contains all requested
            variables. Pass None to disable caching.
        variables: hourly variables to request from Open-Meteo.
        timeout: seconds to wait for the API call.

    Returns:
        Copy of `df` with one new column per variable, joined on the hour.

    Side effects:
        Writes the fetched hourly weather table to `cache_path` if caching
        is enabled and the cache had to be refreshed.
    """
    import json
    from pathlib import Path
    from urllib.parse import urlencode
    from urllib.request import urlopen

    if "datetime" not in df.columns:
        raise KeyError("df must have a 'datetime' column. Run clean_data() first.")

    df = df.copy()
    start_date = df["datetime"].min().date()
    end_date = df["datetime"].max().date()

    cache_path = Path(cache_path) if cache_path else None
    weather = None

    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["time"])
        covers_range = (
            cached["time"].min().date() <= start_date
            and cached["time"].max().date() >= end_date
        )
        has_all_vars = all(v in cached.columns for v in variables)
        if covers_range and has_all_vars:
            weather = cached[["time", *variables]]

    if weather is None:
        params = {
            "latitude": lat,
            "longitude": lng,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(variables),
            "timezone": "GMT",
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
        with urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        if "hourly" not in payload:
            raise RuntimeError(
                f"Open-Meteo response missing 'hourly' block: {payload}"
            )
        weather = pd.DataFrame(payload["hourly"])
        weather["time"] = pd.to_datetime(weather["time"])
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            weather.to_csv(cache_path, index=False)

    df["_join_hour"] = df["datetime"].dt.floor("h")
    df = df.merge(
        weather.rename(columns={"time": "_join_hour"}),
        on="_join_hour",
        how="left",
    )
    df = df.drop(columns="_join_hour")
    return df
