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


def add_lag_features(df):
    """Add per-station lag and rolling features for bike availability.

    Must be called on a dataset that already has 'bikes', 'station_number',
    'datetime', and 'hour' columns (i.e. after clean_data()).

    Lag steps assume 30-minute intervals:
        1  step  =  30 min
        2  steps =  1 hour
        4  steps =  2 hours
        48 steps = 24 hours
       336 steps =  1 week

    Also adds a station×hour historical mean and a 3-hour rolling mean/std.
    Rows where any lag is NaN (start of each station's history) are dropped.
    """
    df = df.copy()
    df = df.sort_values(['station_number', 'datetime']).reset_index(drop=True)

    grp = df.groupby('station_number', observed=True)['bikes']

    df['bikes_lag_1']   = grp.shift(1)
    df['bikes_lag_2']   = grp.shift(2)
    df['bikes_lag_4']   = grp.shift(4)
    df['bikes_lag_48']  = grp.shift(48)
    df['bikes_lag_336'] = grp.shift(336)

    df['bikes_roll3h_mean'] = grp.transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )
    df['bikes_roll3h_std'] = grp.transform(
        lambda x: x.shift(1).rolling(6, min_periods=2).std().fillna(0)
    )

    station_hour_mean = (
        df.groupby(['station_number', 'hour'], observed=True)['bikes']
        .mean()
        .rename('station_hour_mean')
        .reset_index()
    )
    df = df.merge(station_hour_mean, on=['station_number', 'hour'], how='left')

    lag_cols = ['bikes_lag_1', 'bikes_lag_2', 'bikes_lag_4', 'bikes_lag_48', 'bikes_lag_336']
    df = df.dropna(subset=lag_cols)

    # Re-sort by datetime so callers see chronological order (TimeSeriesSplit,
    # is_monotonic_increasing checks). The earlier sort by [station, datetime]
    # was needed only for correct per-station lag computation.
    df = df.sort_values('datetime').reset_index(drop=True)

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

